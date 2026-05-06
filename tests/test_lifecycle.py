"""Tests for execution lifecycle state machine and stop reasons."""

from __future__ import annotations

import pytest

from anycode.core import stop_reason as stop_reasons
from anycode.core.lifecycle import (
    ALL_PHASES,
    TERMINAL_PHASES,
    InvalidPhaseTransitionError,
    LifecycleEmitter,
    LoopDetector,
    fingerprint_call,
    is_valid_transition,
)
from anycode.types import LifecycleEvent, StopReason


def test_all_phases_present() -> None:
    expected = {
        "initialized",
        "planning",
        "executing",
        "observing",
        "verifying",
        "recovering",
        "completed",
        "failed",
        "cancelled",
    }
    assert set(ALL_PHASES) == expected


def test_terminal_phases() -> None:
    assert TERMINAL_PHASES == frozenset({"completed", "failed", "cancelled"})


def test_valid_transitions() -> None:
    assert is_valid_transition("initialized", "executing")
    assert is_valid_transition("executing", "observing")
    assert is_valid_transition("observing", "executing")
    assert is_valid_transition("verifying", "completed")
    assert is_valid_transition("recovering", "executing")


def test_invalid_transitions() -> None:
    assert not is_valid_transition("initialized", "completed")
    assert not is_valid_transition("planning", "observing")
    assert not is_valid_transition("completed", "executing")
    assert not is_valid_transition("failed", "executing")


def test_emitter_initial_state() -> None:
    emitter = LifecycleEmitter(run_id="r1", agent_name="agent")
    assert emitter.phase == "initialized"
    assert len(emitter.events) == 1
    assert emitter.events[0].phase == "initialized"
    assert emitter.events[0].run_id == "r1"
    assert not emitter.is_terminal


def test_emitter_strict_rejects_invalid_transition() -> None:
    emitter = LifecycleEmitter(run_id="r1", agent_name="agent", strict=True)
    with pytest.raises(InvalidPhaseTransitionError):
        emitter.transition("completed")


def test_emitter_non_strict_allows_invalid_transition() -> None:
    emitter = LifecycleEmitter(run_id="r1", agent_name="agent", strict=False)
    event = emitter.transition("completed", stop_reason=stop_reasons.success())
    assert event.phase == "completed"
    assert emitter.is_terminal


def test_emitter_transition_records_event() -> None:
    emitter = LifecycleEmitter(run_id="r1", agent_name="agent")
    emitter.transition("executing")
    emitter.transition("observing", metadata={"tool_calls": 2})
    emitter.transition("completed", stop_reason=stop_reasons.success())

    phases = [e.phase for e in emitter.events]
    assert phases == ["initialized", "executing", "observing", "completed"]
    assert emitter.events[2].metadata == {"tool_calls": 2}
    assert emitter.is_terminal


def test_emitter_listener_invoked() -> None:
    seen: list[LifecycleEvent] = []

    def listener(event: LifecycleEvent) -> None:
        seen.append(event)

    emitter = LifecycleEmitter(run_id="r1", agent_name="agent", listeners=[listener])
    emitter.transition("executing")
    emitter.transition("completed", stop_reason=stop_reasons.success())

    # Listeners do not see the implicit initialized event from construction.
    assert [e.phase for e in seen] == ["executing", "completed"]


def test_emitter_listener_failure_does_not_break_run() -> None:
    def bad_listener(_: LifecycleEvent) -> None:
        raise RuntimeError("boom")

    emitter = LifecycleEmitter(run_id="r1", agent_name="agent", listeners=[bad_listener])
    event = emitter.transition("executing")
    assert event.phase == "executing"


def test_stop_reason_factories_have_stable_codes() -> None:
    assert stop_reasons.success().code == "success"
    assert stop_reasons.max_turns(10).code == "max_turns"
    assert stop_reasons.budget_exceeded("oom").code == "budget_exceeded"
    assert stop_reasons.tool_error("x").code == "tool_error"
    assert stop_reasons.doom_loop("abc", 3).code == "doom_loop"
    assert stop_reasons.user_cancelled().code == "user_cancelled"
    assert stop_reasons.unknown().code == "unknown"


def test_stop_reason_recoverable_flags() -> None:
    assert stop_reasons.success().recoverable is False
    assert stop_reasons.max_turns(5).recoverable is True
    assert stop_reasons.tool_error("x").recoverable is True
    assert stop_reasons.budget_exceeded("x").recoverable is False


def test_stop_reason_is_frozen() -> None:
    reason = StopReason(code="success", message="ok")
    with pytest.raises(Exception):
        reason.message = "changed"  # type: ignore[misc]


def test_fingerprint_call_stable() -> None:
    fp1 = fingerprint_call("bash", {"command": "ls"})
    fp2 = fingerprint_call("bash", {"command": "ls"})
    fp3 = fingerprint_call("bash", {"command": "pwd"})
    assert fp1 == fp2
    assert fp1 != fp3


def test_loop_detector_detects_repeats() -> None:
    detector = LoopDetector(window=4, repeat_threshold=3)
    fp = fingerprint_call("bash", {"command": "ls"})
    detector.record(fp)
    looping, _, _ = detector.is_looping()
    assert not looping
    detector.record(fp)
    detector.record(fp)
    looping, pattern, repeats = detector.is_looping()
    assert looping is True
    assert pattern == fp
    assert repeats >= 3


def test_loop_detector_resets_window() -> None:
    detector = LoopDetector(window=2, repeat_threshold=2)
    fp_a = fingerprint_call("a", {})
    fp_b = fingerprint_call("b", {})
    detector.record(fp_a)
    detector.record(fp_b)
    detector.record(fp_b)
    looping, pattern, _ = detector.is_looping()
    assert looping is True
    assert pattern == fp_b


def test_lifecycle_event_immutable() -> None:
    event = LifecycleEvent(run_id="r", agent_name="a", phase="executing")
    with pytest.raises(Exception):
        event.run_id = "x"  # type: ignore[misc]
