"""Deterministic state, cancellation, retry, and dependency semantics."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from anycode.contracts.models import (
    CONTRACT_SCHEMA_VERSION,
    Cancellation,
    Checkpoint,
    ContractError,
    ContractModel,
    Event,
    Run,
    RunState,
    Task,
    WaitingReason,
    utc_now,
)
from anycode.helpers.uuid7 import uuid7

TERMINAL_STATES: frozenset[RunState] = frozenset({"succeeded", "failed", "canceled", "rejected"})
RUN_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    "accepted": frozenset({"queued", "canceled", "rejected"}),
    "queued": frozenset({"running", "waiting", "failed", "canceled"}),
    "running": frozenset({"waiting", "succeeded", "failed", "canceled"}),
    "waiting": frozenset({"queued", "running", "failed", "canceled"}),
    "succeeded": frozenset(),
    "failed": frozenset(),
    "canceled": frozenset(),
    "rejected": frozenset(),
}

RetryClass = Literal[
    "transient",
    "rate_limited",
    "provider_unavailable",
    "invalid_request",
    "policy_denied",
    "side_effect_unknown",
    "permanent",
]
RETRYABLE_CLASSES: frozenset[RetryClass] = frozenset({"transient", "rate_limited", "provider_unavailable"})


class TransitionResult(ContractModel):
    ok: bool
    run: Run | None = None
    task: Task | None = None
    event: Event | None = None
    error: ContractError | None = None

    @model_validator(mode="after")
    def _validate_result(self) -> TransitionResult:
        if self.ok and self.error is not None:
            raise ValueError("A successful transition cannot contain an error.")
        if not self.ok and self.error is None:
            raise ValueError("A failed transition requires an error.")
        return self


class CancellationResult(ContractModel):
    ok: bool
    outcome: Literal["requested", "acknowledged", "lost_to_completion", "unchanged"]
    run: Run
    event: Event | None = None
    error: ContractError | None = None


class RetryPolicy(ContractModel):
    max_attempts: int = Field(default=3, ge=1)
    allow_provider_switch: bool = False
    compatible_provider_classes: tuple[str, ...] = ()


class RetryDecision(ContractModel):
    retry: bool
    next_attempt: int | None = None
    provider_switch: bool = False
    reason: str


class DependencyDecision(ContractModel):
    state: Literal["queued", "waiting", "failed", "canceled"]
    waiting_reason: WaitingReason | None = None
    error: ContractError | None = None
    artifact_ids: tuple[str, ...] = ()
    uses_partial_artifacts: bool = False


class CheckpointCompatibility(ContractModel):
    compatible: bool
    reader_version: str = CONTRACT_SCHEMA_VERSION
    checkpoint_version: str
    error: ContractError | None = None


class ResumeResult(ContractModel):
    ok: bool
    run: Run | None = None
    event: Event | None = None
    error: ContractError | None = None


def is_valid_run_transition(current: RunState, target: RunState) -> bool:
    return target in RUN_TRANSITIONS[current]


def _event(
    run: Run,
    *,
    event_type: str,
    payload: dict[str, JsonValue],
    sequence: int,
    causation_id: str | None,
    task_id: str | None = None,
    now: datetime,
) -> Event:
    return Event(
        id=str(uuid7()),
        run_id=run.id,
        task_id=task_id,
        sequence=sequence,
        type=event_type,
        payload=payload,
        correlation_id=run.correlation_id,
        causation_id=causation_id,
        generation=run.generation,
        attempt=run.attempt,
        emitted_at=now,
    )


def transition_run(
    run: Run,
    target: RunState,
    *,
    waiting_reason: WaitingReason | None = None,
    error: ContractError | None = None,
    causation_id: str | None = None,
    expected_generation: int | None = None,
    now: datetime | None = None,
) -> TransitionResult:
    if expected_generation is not None and expected_generation != run.generation:
        return TransitionResult(
            ok=False,
            run=run,
            error=ContractError(
                code="stale_generation",
                message=f"Expected generation {expected_generation}, current generation is {run.generation}.",
            ),
        )
    if not is_valid_run_transition(run.state, target):
        return TransitionResult(
            ok=False,
            run=run,
            error=ContractError(code="illegal_transition", message=f"Illegal run transition: {run.state} -> {target}."),
        )
    if target == "waiting" and waiting_reason is None:
        return TransitionResult(
            ok=False,
            run=run,
            error=ContractError(code="waiting_reason_required", message="A waiting transition requires a typed reason."),
        )
    if target != "waiting" and waiting_reason is not None:
        return TransitionResult(
            ok=False,
            run=run,
            error=ContractError(code="unexpected_waiting_reason", message="A waiting reason is valid only for the waiting state."),
        )
    if target in ("failed", "rejected") and error is None:
        return TransitionResult(
            ok=False,
            run=run,
            error=ContractError(code="terminal_error_required", message=f"The {target} state requires a typed error."),
        )

    timestamp = now or utc_now()
    cancellation = run.cancellation
    if target in ("succeeded", "failed") and cancellation.status == "requested":
        cancellation = cancellation.model_copy(update={"status": "lost_to_completion"})
    sequence = run.last_event_sequence + 1
    event = _event(
        run,
        event_type="run.transitioned",
        payload={
            "from": run.state,
            "to": target,
            "waiting_reason": waiting_reason,
            "error": error.model_dump(mode="json") if error is not None else None,
            "cancellation_status": cancellation.status,
        },
        sequence=sequence,
        causation_id=causation_id,
        now=timestamp,
    )
    updated = run.model_copy(
        update={
            "state": target,
            "waiting_reason": waiting_reason,
            "error": error,
            "cancellation": cancellation,
            "updated_at": timestamp,
            "last_event_sequence": sequence,
            "causation_id": causation_id,
        }
    )
    return TransitionResult(ok=True, run=updated, event=event)


def transition_task(
    run: Run,
    task: Task,
    target: RunState,
    *,
    sequence: int,
    waiting_reason: WaitingReason | None = None,
    error: ContractError | None = None,
    causation_id: str | None = None,
    now: datetime | None = None,
) -> TransitionResult:
    if task.run_id != run.id:
        return TransitionResult(
            ok=False,
            task=task,
            error=ContractError(code="run_mismatch", message="Task does not belong to the supplied run."),
        )
    if not is_valid_run_transition(task.state, target):
        return TransitionResult(
            ok=False,
            task=task,
            error=ContractError(code="illegal_transition", message=f"Illegal task transition: {task.state} -> {target}."),
        )
    if target == "waiting" and waiting_reason is None:
        return TransitionResult(
            ok=False,
            task=task,
            error=ContractError(code="waiting_reason_required", message="A waiting transition requires a typed reason."),
        )
    if target != "waiting" and waiting_reason is not None:
        return TransitionResult(
            ok=False,
            task=task,
            error=ContractError(code="unexpected_waiting_reason", message="A waiting reason is valid only for the waiting state."),
        )
    if target in ("failed", "rejected") and error is None:
        return TransitionResult(
            ok=False,
            task=task,
            error=ContractError(code="terminal_error_required", message=f"The {target} state requires a typed error."),
        )

    timestamp = now or utc_now()
    event = _event(
        run,
        event_type="task.transitioned",
        payload={
            "from": task.state,
            "to": target,
            "waiting_reason": waiting_reason,
            "error": error.model_dump(mode="json") if error is not None else None,
        },
        sequence=sequence,
        causation_id=causation_id,
        task_id=task.id,
        now=timestamp,
    )
    updated = task.model_copy(
        update={
            "state": target,
            "waiting_reason": waiting_reason,
            "error": error,
            "updated_at": timestamp,
            "causation_id": causation_id,
        }
    )
    return TransitionResult(ok=True, task=updated, event=event)


def request_cancellation(
    run: Run,
    *,
    reason: str,
    causation_id: str | None = None,
    now: datetime | None = None,
) -> CancellationResult:
    if run.cancellation.status != "none":
        return CancellationResult(ok=True, outcome=run.cancellation.status, run=run)

    timestamp = now or utc_now()
    if run.state in TERMINAL_STATES:
        cancellation = Cancellation(status="lost_to_completion", requested_at=timestamp, reason=reason)
        request_outcome: Literal["requested", "lost_to_completion"] = "lost_to_completion"
        event_type = "cancellation.lost"
    else:
        cancellation = Cancellation(status="requested", requested_at=timestamp, reason=reason)
        request_outcome = "requested"
        event_type = "cancellation.requested"
    sequence = run.last_event_sequence + 1
    event = _event(
        run,
        event_type=event_type,
        payload={"reason": reason, "status": cancellation.status},
        sequence=sequence,
        causation_id=causation_id,
        now=timestamp,
    )
    updated = run.model_copy(
        update={"cancellation": cancellation, "updated_at": timestamp, "last_event_sequence": sequence, "causation_id": causation_id}
    )
    return CancellationResult(ok=True, outcome=request_outcome, run=updated, event=event)


def acknowledge_cancellation(
    run: Run,
    *,
    causation_id: str | None = None,
    now: datetime | None = None,
) -> CancellationResult:
    if run.cancellation.status == "acknowledged":
        return CancellationResult(ok=True, outcome="unchanged", run=run)
    if run.cancellation.status == "lost_to_completion" or run.state in TERMINAL_STATES:
        lost = run.cancellation.model_copy(update={"status": "lost_to_completion"})
        return CancellationResult(ok=True, outcome="lost_to_completion", run=run.model_copy(update={"cancellation": lost}))
    if run.cancellation.status != "requested":
        return CancellationResult(
            ok=False,
            outcome="unchanged",
            run=run,
            error=ContractError(code="cancellation_not_requested", message="Cancellation must be requested before acknowledgement."),
        )

    timestamp = now or utc_now()
    cancellation = run.cancellation.model_copy(update={"status": "acknowledged", "acknowledged_at": timestamp})
    sequence = run.last_event_sequence + 1
    event = _event(
        run,
        event_type="cancellation.acknowledged",
        payload={"from": run.state, "to": "canceled", "status": "acknowledged"},
        sequence=sequence,
        causation_id=causation_id,
        now=timestamp,
    )
    updated = run.model_copy(
        update={
            "state": "canceled",
            "waiting_reason": None,
            "cancellation": cancellation,
            "updated_at": timestamp,
            "last_event_sequence": sequence,
            "causation_id": causation_id,
        }
    )
    return CancellationResult(ok=True, outcome="acknowledged", run=updated, event=event)


def decide_retry(
    classification: RetryClass,
    *,
    attempt: int,
    policy: RetryPolicy,
    current_provider: str,
    candidate_provider: str | None = None,
    current_compatibility_class: str | None = None,
    candidate_compatibility_class: str | None = None,
) -> RetryDecision:
    if classification not in RETRYABLE_CLASSES:
        return RetryDecision(retry=False, reason=f"{classification} is not retryable.")
    if attempt >= policy.max_attempts:
        return RetryDecision(retry=False, reason="Retry attempts are exhausted.")

    selected = candidate_provider or current_provider
    provider_switch = selected != current_provider
    if provider_switch and not policy.allow_provider_switch:
        return RetryDecision(retry=False, reason="Provider switching is disabled.")
    if provider_switch and current_compatibility_class != candidate_compatibility_class:
        return RetryDecision(retry=False, reason="Candidate provider is not in the same fallback compatibility class.")
    if provider_switch and policy.compatible_provider_classes and candidate_compatibility_class not in policy.compatible_provider_classes:
        return RetryDecision(retry=False, reason="Candidate compatibility class is not allowed by policy.")
    return RetryDecision(retry=True, next_attempt=attempt + 1, provider_switch=provider_switch, reason="Retry is allowed.")


def evaluate_dependencies(task: Task, dependencies: tuple[Task, ...]) -> DependencyDecision:
    expected = set(task.dependencies)
    supplied = {dependency.id for dependency in dependencies}
    if expected != supplied:
        return DependencyDecision(state="waiting", waiting_reason="dependency")

    artifacts = tuple(artifact_id for dependency in dependencies for artifact_id in dependency.produced_artifact_ids)
    failed = tuple(dependency for dependency in dependencies if dependency.state in ("failed", "rejected", "canceled"))
    if failed:
        if task.allow_partial_dependency_artifacts and all(dependency.produced_artifact_ids for dependency in failed):
            return DependencyDecision(state="queued", artifact_ids=artifacts, uses_partial_artifacts=True)
        if all(dependency.state == "canceled" for dependency in failed):
            return DependencyDecision(
                state="canceled",
                error=ContractError(code="dependency_canceled", message="A required dependency was canceled."),
                artifact_ids=artifacts,
            )
        return DependencyDecision(
            state="failed",
            error=ContractError(code="dependency_failed", message="A required dependency failed or was rejected."),
            artifact_ids=artifacts,
        )
    if any(dependency.state != "succeeded" for dependency in dependencies):
        return DependencyDecision(state="waiting", waiting_reason="dependency", artifact_ids=artifacts)
    return DependencyDecision(state="queued", artifact_ids=artifacts)


def check_checkpoint_compatibility(checkpoint: Checkpoint) -> CheckpointCompatibility:
    compatible = checkpoint.schema_version == CONTRACT_SCHEMA_VERSION and checkpoint.generation == checkpoint.run.generation
    error = None
    if not compatible:
        error = ContractError(code="incompatible_checkpoint", message="Checkpoint schema or generation is incompatible with the run.")
    return CheckpointCompatibility(
        compatible=compatible,
        checkpoint_version=checkpoint.schema_version,
        error=error,
    )


def start_new_generation(
    run: Run,
    checkpoint: Checkpoint,
    *,
    causation_id: str | None = None,
    now: datetime | None = None,
) -> ResumeResult:
    compatibility = check_checkpoint_compatibility(checkpoint)
    if checkpoint.run_id != run.id or checkpoint.generation != run.generation or not compatibility.compatible:
        return ResumeResult(
            ok=False,
            error=ContractError(code="incompatible_checkpoint", message="Checkpoint does not match the current run generation."),
        )
    if run.state not in TERMINAL_STATES:
        return ResumeResult(
            ok=False,
            error=ContractError(code="run_not_terminal", message="A new generation can start only after a terminal generation."),
        )

    timestamp = now or utc_now()
    sequence = run.last_event_sequence + 1
    next_run = run.model_copy(
        update={
            "state": "accepted",
            "generation": run.generation + 1,
            "attempt": 1,
            "waiting_reason": None,
            "cancellation": Cancellation(),
            "error": None,
            "updated_at": timestamp,
            "last_event_sequence": sequence,
            "causation_id": causation_id,
        }
    )
    event = Event(
        id=str(uuid7()),
        run_id=run.id,
        sequence=sequence,
        type="run.generation_started",
        payload={"from_generation": run.generation, "to_generation": next_run.generation, "checkpoint_id": checkpoint.id},
        correlation_id=run.correlation_id,
        causation_id=causation_id,
        generation=next_run.generation,
        attempt=1,
        emitted_at=timestamp,
    )
    return ResumeResult(ok=True, run=next_run, event=event)
