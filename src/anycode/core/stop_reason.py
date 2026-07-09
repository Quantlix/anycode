"""Structured stop-reason factories for terminal and recoverable run outcomes."""

from __future__ import annotations

from anycode.types import StopReason


def success(message: str = "Run completed successfully.") -> StopReason:
    return StopReason(code="success", message=message, recoverable=False)


def max_turns(turn_limit: int) -> StopReason:
    return StopReason(
        code="max_turns",
        message=f"Reached maximum turn limit of {turn_limit}.",
        recoverable=True,
    )


def budget_exceeded(reason: str) -> StopReason:
    return StopReason(code="budget_exceeded", message=reason, recoverable=False)


def context_pressure(message: str) -> StopReason:
    return StopReason(code="context_pressure", message=message, recoverable=True)


def tool_error(message: str) -> StopReason:
    return StopReason(code="tool_error", message=message, recoverable=True)


def verification_failed(message: str) -> StopReason:
    return StopReason(code="verification_failed", message=message, recoverable=True)


def blocked_dependency(message: str) -> StopReason:
    return StopReason(code="blocked_dependency", message=message, recoverable=False)


def user_cancelled(message: str = "Run cancelled by user or policy.") -> StopReason:
    return StopReason(code="user_cancelled", message=message, recoverable=False)


def doom_loop(pattern: str, repeats: int) -> StopReason:
    return StopReason(
        code="doom_loop",
        message=f"Detected repeated tool-call pattern '{pattern}' x{repeats}.",
        recoverable=True,
    )


def provider_unavailable(message: str) -> StopReason:
    return StopReason(code="provider_unavailable", message=message, recoverable=True)


def unknown(message: str = "Run ended for unknown reasons.") -> StopReason:
    return StopReason(code="unknown", message=message, recoverable=False)
