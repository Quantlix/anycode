"""Event ordering validation and two independent run projections."""

from __future__ import annotations

from pydantic import Field

from anycode.contracts.models import CancellationStatus, ContractError, ContractModel, Event, RunState


class EventStreamValidation(ContractModel):
    valid: bool
    cursor: int = Field(ge=0)
    error: ContractError | None = None


class RunProjection(ContractModel):
    run_id: str
    state: RunState | None = None
    generation: int = Field(default=1, ge=1)
    cursor: int = Field(default=0, ge=0)
    task_states: dict[str, RunState] = Field(default_factory=dict)
    artifact_ids: tuple[str, ...] = ()
    cancellation_status: CancellationStatus = "none"
    applied_event_ids: tuple[str, ...] = ()


class ProjectionResult(ContractModel):
    ok: bool
    projection: RunProjection | None = None
    error: ContractError | None = None


def validate_event_stream(events: tuple[Event, ...] | list[Event]) -> EventStreamValidation:
    if not events:
        return EventStreamValidation(valid=True, cursor=0)
    run_id = events[0].run_id
    seen_ids: set[str] = set()
    expected_sequence = 1
    for event in events:
        if event.run_id != run_id:
            return EventStreamValidation(
                valid=False,
                cursor=expected_sequence - 1,
                error=ContractError(code="mixed_run_stream", message="An event stream cannot contain multiple run ids."),
            )
        if event.sequence != expected_sequence:
            return EventStreamValidation(
                valid=False,
                cursor=expected_sequence - 1,
                error=ContractError(
                    code="event_sequence_gap",
                    message=f"Expected event sequence {expected_sequence}, received {event.sequence}.",
                ),
            )
        if event.id in seen_ids:
            return EventStreamValidation(
                valid=False,
                cursor=expected_sequence - 1,
                error=ContractError(code="duplicate_event", message=f"Event id {event.id} was delivered more than once."),
            )
        if event.causation_id is not None and event.causation_id not in seen_ids:
            return EventStreamValidation(
                valid=False,
                cursor=expected_sequence - 1,
                error=ContractError(code="unknown_causation", message="An event cannot reference a later or absent causal event."),
            )
        seen_ids.add(event.id)
        expected_sequence += 1
    return EventStreamValidation(valid=True, cursor=len(events))


def _payload_state(event: Event, key: str) -> RunState | None:
    value = event.payload.get(key)
    if value in {"accepted", "queued", "running", "waiting", "succeeded", "failed", "canceled", "rejected"}:
        return value  # type: ignore[return-value]
    return None


def _payload_text(event: Event, key: str) -> str | None:
    value = event.payload.get(key)
    return value if isinstance(value, str) and value else None


class IncrementalRunProjector:
    """Mutable incremental projector used by streaming consumers."""

    def __init__(self, run_id: str) -> None:
        self._projection = RunProjection(run_id=run_id)

    @property
    def projection(self) -> RunProjection:
        return self._projection

    def apply(self, event: Event) -> ProjectionResult:
        expected = self._projection.cursor + 1
        if event.run_id != self._projection.run_id:
            return ProjectionResult(
                ok=False,
                projection=self._projection,
                error=ContractError(code="run_mismatch", message="Event belongs to a different run."),
            )
        if event.sequence != expected:
            return ProjectionResult(
                ok=False,
                projection=self._projection,
                error=ContractError(code="event_sequence_gap", message=f"Expected sequence {expected}, received {event.sequence}."),
            )
        if event.id in self._projection.applied_event_ids:
            return ProjectionResult(
                ok=False,
                projection=self._projection,
                error=ContractError(code="duplicate_event", message="Event was already applied."),
            )

        state = self._projection.state
        generation = max(self._projection.generation, event.generation)
        task_states = dict(self._projection.task_states)
        artifact_ids = list(self._projection.artifact_ids)
        cancellation_status = self._projection.cancellation_status

        if event.type == "run.admitted":
            state = "accepted"
        elif event.type == "run.transitioned":
            state = _payload_state(event, "to") or state
            raw_cancellation = event.payload.get("cancellation_status")
            if raw_cancellation in {"none", "requested", "acknowledged", "lost_to_completion"}:
                cancellation_status = raw_cancellation  # type: ignore[assignment]
        elif event.type == "run.generation_started":
            state = "accepted"
        elif event.type == "task.transitioned" and event.task_id is not None:
            next_state = _payload_state(event, "to")
            if next_state is not None:
                task_states[event.task_id] = next_state
        elif event.type == "artifact.committed":
            artifact_id = _payload_text(event, "artifact_id")
            if artifact_id is not None and artifact_id not in artifact_ids:
                artifact_ids.append(artifact_id)
        elif event.type == "cancellation.requested":
            cancellation_status = "requested"
        elif event.type == "cancellation.acknowledged":
            cancellation_status = "acknowledged"
            state = "canceled"
        elif event.type == "cancellation.lost":
            cancellation_status = "lost_to_completion"

        self._projection = self._projection.model_copy(
            update={
                "state": state,
                "generation": generation,
                "cursor": event.sequence,
                "task_states": task_states,
                "artifact_ids": tuple(artifact_ids),
                "cancellation_status": cancellation_status,
                "applied_event_ids": (*self._projection.applied_event_ids, event.id),
            }
        )
        return ProjectionResult(ok=True, projection=self._projection)


def project_run(events: tuple[Event, ...] | list[Event]) -> ProjectionResult:
    """Pure batch projector implemented independently from the incremental class."""
    validation = validate_event_stream(events)
    if not validation.valid:
        return ProjectionResult(ok=False, error=validation.error)
    if not events:
        return ProjectionResult(
            ok=False,
            error=ContractError(code="empty_event_stream", message="A run projection requires at least one event."),
        )

    run_state: RunState | None = None
    cancellation: CancellationStatus = "none"
    tasks: dict[str, RunState] = {}
    artifacts: list[str] = []
    for event in events:
        if event.type == "run.admitted":
            run_state = "accepted"
        if event.type == "run.transitioned":
            run_state = _payload_state(event, "to") or run_state
            status = event.payload.get("cancellation_status")
            if status in {"none", "requested", "acknowledged", "lost_to_completion"}:
                cancellation = status  # type: ignore[assignment]
        if event.type == "run.generation_started":
            run_state = "accepted"
        if event.type == "task.transitioned" and event.task_id:
            task_state = _payload_state(event, "to")
            if task_state is not None:
                tasks = {**tasks, event.task_id: task_state}
        if event.type == "artifact.committed":
            artifact_id = _payload_text(event, "artifact_id")
            if artifact_id and artifact_id not in artifacts:
                artifacts = [*artifacts, artifact_id]
        if event.type == "cancellation.requested":
            cancellation = "requested"
        if event.type == "cancellation.acknowledged":
            cancellation = "acknowledged"
            run_state = "canceled"
        if event.type == "cancellation.lost":
            cancellation = "lost_to_completion"

    return ProjectionResult(
        ok=True,
        projection=RunProjection(
            run_id=events[0].run_id,
            state=run_state,
            generation=max(event.generation for event in events),
            cursor=events[-1].sequence,
            task_states=tasks,
            artifact_ids=tuple(artifacts),
            cancellation_status=cancellation,
            applied_event_ids=tuple(event.id for event in events),
        ),
    )
