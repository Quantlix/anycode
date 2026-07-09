"""Deterministic failure categorization.

The taxonomy is rule-based on purpose: a small, versioned set of categories is
easier for the evolution loop to reason over than open-ended LLM summaries.
Callers can override categorization by passing custom ``rules`` to
:func:`categorize_event` and :func:`categorize_run`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from anycode.types import (
    AgentRunResult,
    FailureCategory,
    LifecycleEvent,
    StopReason,
    ToolCallRecord,
    VerificationResult,
)

DEFAULT_TAXONOMY_VERSION = "1.0"


_STOP_REASON_MAP: Mapping[str, FailureCategory] = {
    "success": FailureCategory.SUCCESS,
    "max_turns": FailureCategory.EARLY_STOPPING,
    "budget_exceeded": FailureCategory.BUDGET_EXCEEDED,
    "context_pressure": FailureCategory.CONTEXT_LOSS,
    "tool_error": FailureCategory.TOOL_RUNTIME_ERROR,
    "verification_failed": FailureCategory.VERIFICATION_FAILURE,
    "blocked_dependency": FailureCategory.POLICY_BLOCKED,
    "user_cancelled": FailureCategory.POLICY_BLOCKED,
    "doom_loop": FailureCategory.EARLY_STOPPING,
    "unknown": FailureCategory.UNKNOWN,
}


_TOOL_ARG_HINTS: tuple[str, ...] = (
    "invalid argument",
    "validation error",
    "missing required",
    "expected type",
    "schema",
    "must be",
    "is not a valid",
)


def _looks_like_argument_error(payload: str) -> bool:
    lowered = payload.lower()
    return any(hint in lowered for hint in _TOOL_ARG_HINTS)


def categorize_stop_reason(stop_reason: StopReason | None) -> FailureCategory:
    if stop_reason is None:
        return FailureCategory.UNKNOWN
    return _STOP_REASON_MAP.get(stop_reason.code, FailureCategory.UNKNOWN)


def categorize_event(event: Any) -> FailureCategory:
    """Categorize a single event-like value.

    Accepts :class:`LifecycleEvent`, :class:`ToolCallRecord`,
    :class:`VerificationResult`, or :class:`StopReason`. Anything else returns
    :attr:`FailureCategory.UNKNOWN` so callers can chain rules safely.
    """

    if isinstance(event, StopReason):
        return categorize_stop_reason(event)
    if isinstance(event, VerificationResult):
        if event.passed:
            return FailureCategory.SUCCESS
        return FailureCategory.VERIFICATION_FAILURE
    if isinstance(event, ToolCallRecord):
        output = (event.output or "").strip()
        if not output:
            return FailureCategory.TOOL_RUNTIME_ERROR
        if _looks_like_argument_error(output):
            return FailureCategory.TOOL_ARGUMENT_ERROR
        if "error" in output.lower() or "exception" in output.lower():
            return FailureCategory.TOOL_RUNTIME_ERROR
        return FailureCategory.SUCCESS
    if isinstance(event, LifecycleEvent):
        if event.stop_reason is not None:
            return categorize_stop_reason(event.stop_reason)
        if event.phase == "failed":
            return FailureCategory.UNKNOWN
        return FailureCategory.SUCCESS
    return FailureCategory.UNKNOWN


def categorize_run(result: AgentRunResult | None) -> FailureCategory:
    """Return the dominant failure category for an agent run."""

    if result is None:
        return FailureCategory.UNKNOWN
    if result.stop_reason is not None and result.stop_reason.code != "success":
        return categorize_stop_reason(result.stop_reason)
    failed_sensors = [vr for vr in result.verification_results if not vr.passed]
    if failed_sensors:
        return FailureCategory.VERIFICATION_FAILURE
    error_tools = [tc for tc in result.tool_calls if categorize_event(tc) != FailureCategory.SUCCESS]
    if error_tools:
        # Surface the most specific category if any argument errors are present.
        if any(categorize_event(tc) == FailureCategory.TOOL_ARGUMENT_ERROR for tc in error_tools):
            return FailureCategory.TOOL_ARGUMENT_ERROR
        return FailureCategory.TOOL_RUNTIME_ERROR
    return FailureCategory.SUCCESS


__all__ = [
    "DEFAULT_TAXONOMY_VERSION",
    "categorize_event",
    "categorize_run",
    "categorize_stop_reason",
]
