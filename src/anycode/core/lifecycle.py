"""Execution lifecycle state machine, transitions, and emitter for agent runs."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any, get_args

from anycode.types import ExecutionPhase, LifecycleEvent, StopReason

# Allowed transitions between phases. Strict by default; orchestrators can bypass via emit_phase.
_VALID_TRANSITIONS: dict[ExecutionPhase, set[ExecutionPhase]] = {
    "initialized": {"planning", "executing", "cancelled", "failed"},
    "planning": {"executing", "cancelled", "failed"},
    "executing": {"observing", "verifying", "recovering", "completed", "failed", "cancelled"},
    "observing": {"executing", "verifying", "recovering", "completed", "failed", "cancelled"},
    "verifying": {"completed", "executing", "recovering", "failed", "cancelled"},
    "recovering": {"executing", "observing", "completed", "failed", "cancelled"},
    "completed": set(),
    "failed": set(),
    "cancelled": set(),
}

TERMINAL_PHASES: frozenset[ExecutionPhase] = frozenset({"completed", "failed", "cancelled"})

ALL_PHASES: tuple[ExecutionPhase, ...] = get_args(ExecutionPhase)


class InvalidPhaseTransitionError(RuntimeError):
    """Raised when a lifecycle transition violates the configured state machine."""


def is_valid_transition(current: ExecutionPhase, target: ExecutionPhase) -> bool:
    return target in _VALID_TRANSITIONS.get(current, set())


# Default fingerprint window for repeated tool-call detection.
_FINGERPRINT_WINDOW = 4
_FINGERPRINT_REPEAT_THRESHOLD = 3


def fingerprint_call(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Produce a stable hash for a tool call so repeats can be detected cheaply."""
    payload = f"{tool_name}:{sorted(tool_input.items())}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


class LoopDetector:
    """Detects repeated tool-call fingerprints that indicate a doom loop."""

    def __init__(
        self,
        window: int = _FINGERPRINT_WINDOW,
        repeat_threshold: int = _FINGERPRINT_REPEAT_THRESHOLD,
    ) -> None:
        self._window = window
        self._threshold = repeat_threshold
        self._recent: list[str] = []

    def record(self, fingerprint: str) -> None:
        self._recent.append(fingerprint)
        if len(self._recent) > self._window:
            self._recent = self._recent[-self._window :]

    def is_looping(self) -> tuple[bool, str | None, int]:
        if not self._recent:
            return False, None, 0
        last = self._recent[-1]
        repeats = sum(1 for f in self._recent if f == last)
        if repeats >= self._threshold:
            return True, last, repeats
        return False, None, repeats

    def reset(self) -> None:
        self._recent.clear()

    def export_window(self) -> tuple[str, ...]:
        """Capture the fingerprint window for durable checkpoints."""
        return tuple(self._recent)

    def restore_window(self, window: tuple[str, ...] | list[str]) -> None:
        """Restore the fingerprint window from a durable checkpoint."""
        self._recent = list(window)[-self._window :]


LifecycleListener = Callable[[LifecycleEvent], None]


class LifecycleEmitter:
    """Tracks the current phase and broadcasts lifecycle events to listeners."""

    def __init__(
        self,
        run_id: str,
        agent_name: str,
        *,
        task_id: str | None = None,
        strict: bool = True,
        listeners: list[LifecycleListener] | None = None,
    ) -> None:
        self._run_id = run_id
        self._agent_name = agent_name
        self._task_id = task_id
        self._strict = strict
        self._listeners: list[LifecycleListener] = list(listeners or [])
        self._phase: ExecutionPhase = "initialized"
        self._events: list[LifecycleEvent] = []
        # Record the implicit initialized event so consumers see the full trail.
        self._events.append(
            LifecycleEvent(
                run_id=run_id,
                agent_name=agent_name,
                task_id=task_id,
                phase="initialized",
            )
        )

    @property
    def phase(self) -> ExecutionPhase:
        return self._phase

    @property
    def events(self) -> list[LifecycleEvent]:
        return list(self._events)

    @property
    def is_terminal(self) -> bool:
        return self._phase in TERMINAL_PHASES

    def add_listener(self, listener: LifecycleListener) -> None:
        self._listeners.append(listener)

    def transition(
        self,
        target: ExecutionPhase,
        *,
        stop_reason: StopReason | None = None,
        metadata: dict[str, str | int | float | bool] | None = None,
    ) -> LifecycleEvent:
        if not is_valid_transition(self._phase, target):
            if self._strict:
                raise InvalidPhaseTransitionError(f"Invalid transition: {self._phase} -> {target} (run_id={self._run_id})")
        event = LifecycleEvent(
            run_id=self._run_id,
            agent_name=self._agent_name,
            task_id=self._task_id,
            phase=target,
            stop_reason=stop_reason,
            metadata=dict(metadata or {}),
        )
        self._phase = target
        self._events.append(event)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:  # noqa: BLE001 - listener failures must not break the run
                continue
        return event
