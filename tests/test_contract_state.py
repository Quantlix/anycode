from datetime import UTC, datetime

import pytest

from anycode.contracts import (
    RUN_TRANSITIONS,
    Checkpoint,
    ContractError,
    RetryPolicy,
    Run,
    Task,
    acknowledge_cancellation,
    decide_retry,
    evaluate_dependencies,
    is_valid_run_transition,
    request_cancellation,
    start_new_generation,
    transition_run,
)
from anycode.contracts.models import RunState

NOW = datetime(2026, 1, 1, tzinfo=UTC)
STATES: tuple[RunState, ...] = ("accepted", "queued", "running", "waiting", "succeeded", "failed", "canceled", "rejected")


def _run(state: RunState = "accepted", *, sequence: int = 0) -> Run:
    return Run(
        id="run-1",
        state=state,
        correlation_id="corr-1",
        waiting_reason="dependency" if state == "waiting" else None,
        last_event_sequence=sequence,
        created_at=NOW,
        updated_at=NOW,
    )


@pytest.mark.parametrize(("current", "target"), [(current, target) for current in STATES for target in STATES])
def test_transition_matrix_is_exhaustive(current: RunState, target: RunState) -> None:
    assert is_valid_run_transition(current, target) is (target in RUN_TRANSITIONS[current])


def test_transitions_require_typed_waits_errors_and_current_generation() -> None:
    run = _run("queued")

    assert transition_run(run, "waiting").error.code == "waiting_reason_required"  # type: ignore[union-attr]
    assert transition_run(run, "failed").error.code == "terminal_error_required"  # type: ignore[union-attr]
    assert transition_run(run, "running", expected_generation=2).error.code == "stale_generation"  # type: ignore[union-attr]

    waiting = transition_run(run, "waiting", waiting_reason="capacity", now=NOW)
    assert waiting.ok and waiting.run is not None and waiting.run.waiting_reason == "capacity"
    assert waiting.event is not None and waiting.event.sequence == 1


def test_cancellation_before_start_acknowledges_once_and_rejects_late_results() -> None:
    requested = request_cancellation(_run(), reason="user request", now=NOW)
    repeated = request_cancellation(requested.run, reason="duplicate", now=NOW)
    acknowledged = acknowledge_cancellation(requested.run, now=NOW)
    repeated_ack = acknowledge_cancellation(acknowledged.run, now=NOW)

    assert requested.outcome == "requested" and requested.run.state == "accepted"
    assert repeated.outcome == "requested" and repeated.event is None
    assert acknowledged.outcome == "acknowledged" and acknowledged.run.state == "canceled"
    assert repeated_ack.outcome == "unchanged" and repeated_ack.event is None
    assert not transition_run(acknowledged.run, "succeeded").ok


def test_terminal_completion_wins_a_race_with_requested_cancellation() -> None:
    requested = request_cancellation(_run("running"), reason="too late", now=NOW)
    completed = transition_run(requested.run, "succeeded", now=NOW)
    assert completed.run is not None

    late_ack = acknowledge_cancellation(completed.run, now=NOW)
    terminal_request = request_cancellation(_run("succeeded"), reason="already done", now=NOW)

    assert completed.run.state == "succeeded"
    assert completed.run.cancellation.status == "lost_to_completion"
    assert late_ack.outcome == "lost_to_completion" and late_ack.run.state == "succeeded"
    assert terminal_request.outcome == "lost_to_completion" and terminal_request.run.state == "succeeded"


def test_retry_policy_requires_retryable_class_budget_and_provider_compatibility() -> None:
    policy = RetryPolicy(max_attempts=3, allow_provider_switch=True, compatible_provider_classes=("chat",))

    allowed = decide_retry(
        "provider_unavailable",
        attempt=1,
        policy=policy,
        current_provider="provider-a",
        candidate_provider="provider-b",
        current_compatibility_class="chat",
        candidate_compatibility_class="chat",
    )
    incompatible = decide_retry(
        "provider_unavailable",
        attempt=1,
        policy=policy,
        current_provider="provider-a",
        candidate_provider="provider-b",
        current_compatibility_class="chat",
        candidate_compatibility_class="embedding",
    )

    assert allowed.retry and allowed.next_attempt == 2 and allowed.provider_switch
    assert not incompatible.retry
    assert not decide_retry("permanent", attempt=1, policy=policy, current_provider="provider-a").retry
    assert not decide_retry("transient", attempt=3, policy=policy, current_provider="provider-a").retry


def test_dependency_failure_can_use_explicit_partial_artifacts() -> None:
    failed = Task(
        id="dependency",
        run_id="run-1",
        state="failed",
        title="dependency",
        correlation_id="corr-1",
        produced_artifact_ids=("partial-1",),
        error=ContractError(code="failed", message="failed"),
    )
    strict = Task(id="strict", run_id="run-1", title="strict", correlation_id="corr-1", dependencies=("dependency",))
    partial = strict.model_copy(update={"id": "partial", "allow_partial_dependency_artifacts": True})

    assert evaluate_dependencies(strict, (failed,)).state == "failed"
    decision = evaluate_dependencies(partial, (failed,))
    assert decision.state == "queued" and decision.uses_partial_artifacts
    assert decision.artifact_ids == ("partial-1",)
    assert evaluate_dependencies(strict, ()).state == "waiting"


def test_terminal_checkpoint_starts_a_new_generation() -> None:
    terminal = _run("succeeded", sequence=4)
    checkpoint = Checkpoint(
        id="checkpoint-1",
        run_id=terminal.id,
        event_cursor=4,
        generation=1,
        attempt=1,
        correlation_id=terminal.correlation_id,
        run=terminal,
        created_at=NOW,
    )

    resumed = start_new_generation(terminal, checkpoint, now=NOW)
    active = start_new_generation(_run("running", sequence=4), checkpoint, now=NOW)

    assert resumed.ok and resumed.run is not None and resumed.run.generation == 2
    assert resumed.run.state == "accepted" and resumed.run.last_event_sequence == 5
    assert resumed.event is not None and resumed.event.generation == 2
    assert not active.ok and active.error is not None and active.error.code == "run_not_terminal"


@pytest.mark.parametrize("state", ("succeeded", "failed", "canceled", "rejected"))
def test_cancellation_request_has_one_deterministic_outcome_for_every_terminal_state(state: RunState) -> None:
    terminal = _run(state)
    requested = request_cancellation(terminal, reason="terminal race", now=NOW)
    repeated = request_cancellation(requested.run, reason="duplicate", now=NOW)

    assert requested.outcome == "lost_to_completion"
    assert requested.run.state == state
    assert requested.run.cancellation.status == "lost_to_completion"
    assert requested.event is not None and requested.event.type == "cancellation.lost"
    assert repeated.outcome == "lost_to_completion" and repeated.event is None
