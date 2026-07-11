"""Structured log events for agent lifecycle transitions."""

from __future__ import annotations

import time
from collections import deque
from datetime import UTC, datetime
from typing import Any

from anycode.constants import (
    TEL_EVENT_AGENT_COMPLETE,
    TEL_EVENT_AGENT_ERROR,
    TEL_EVENT_AGENT_START,
    TEL_EVENT_BUDGET_EXHAUSTED,
    TEL_EVENT_BUDGET_WARNING,
    TEL_EVENT_LLM_CALL_COMPLETE,
    TEL_EVENT_LLM_CALL_START,
    TEL_EVENT_TOOL_COMPLETE,
    TEL_EVENT_TOOL_START,
    TEL_EVENT_TURN_COMPLETE,
    TEL_EVENT_TURN_START,
)
from anycode.security.redaction import redact_sensitive


class TelemetryEvent:
    """Represents a structured lifecycle event."""

    def __init__(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.name = name
        self.attributes = attributes or {}
        self.timestamp = time.monotonic()
        self.observed_at = datetime.now(UTC)

    def to_dict(self, *, redact_sensitive_data: bool = True) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "attributes": self.attributes,
            "timestamp": self.timestamp,
            "observed_at": self.observed_at.isoformat(),
        }
        return redact_sensitive(payload) if redact_sensitive_data else payload


class EventEmitter:
    """Collects and dispatches structured lifecycle events."""

    def __init__(self, enabled: bool = False, *, max_events: int = 10_000) -> None:
        if max_events < 1:
            raise ValueError(f"max_events must be >= 1, received {max_events}")
        self._enabled = enabled
        self._events: deque[TelemetryEvent] = deque(maxlen=max_events)
        self._dropped_events = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def events(self) -> list[TelemetryEvent]:
        return list(self._events)

    @property
    def dropped_events(self) -> int:
        return self._dropped_events

    def emit(self, name: str, attributes: dict[str, Any] | None = None) -> TelemetryEvent | None:
        if not self._enabled:
            return None
        event = TelemetryEvent(name, attributes)
        if len(self._events) == self._events.maxlen:
            self._dropped_events += 1
        self._events.append(event)
        return event

    def agent_start(self, agent_name: str, model: str) -> None:
        self.emit(TEL_EVENT_AGENT_START, {"agent_name": agent_name, "model": model})

    def agent_complete(self, agent_name: str, turns: int, tokens_used: int) -> None:
        self.emit(TEL_EVENT_AGENT_COMPLETE, {"agent_name": agent_name, "turns": turns, "tokens_used": tokens_used})

    def agent_error(self, agent_name: str, error: str) -> None:
        self.emit(TEL_EVENT_AGENT_ERROR, {"agent_name": agent_name, "error": error})

    def turn_start(self, agent_name: str, turn_number: int) -> None:
        self.emit(TEL_EVENT_TURN_START, {"agent_name": agent_name, "turn_number": turn_number})

    def turn_complete(self, agent_name: str, turn_number: int, token_input: int, token_output: int) -> None:
        self.emit(
            TEL_EVENT_TURN_COMPLETE,
            {"agent_name": agent_name, "turn_number": turn_number, "token_input": token_input, "token_output": token_output},
        )

    def tool_start(self, agent_name: str, tool_name: str) -> None:
        self.emit(TEL_EVENT_TOOL_START, {"agent_name": agent_name, "tool_name": tool_name})

    def tool_complete(self, agent_name: str, tool_name: str, duration_ms: float, is_error: bool) -> None:
        self.emit(
            TEL_EVENT_TOOL_COMPLETE,
            {"agent_name": agent_name, "tool_name": tool_name, "duration_ms": duration_ms, "is_error": is_error},
        )

    def llm_call_start(self, agent_name: str, model: str) -> None:
        self.emit(TEL_EVENT_LLM_CALL_START, {"agent_name": agent_name, "model": model})

    def llm_call_complete(self, agent_name: str, model: str, input_tokens: int, output_tokens: int, duration_ms: float) -> None:
        self.emit(
            TEL_EVENT_LLM_CALL_COMPLETE,
            {"agent_name": agent_name, "model": model, "input_tokens": input_tokens, "output_tokens": output_tokens, "duration_ms": duration_ms},
        )

    def budget_warning(self, agent_name: str, resource: str, used: float, limit: float) -> None:
        self.emit(TEL_EVENT_BUDGET_WARNING, {"agent_name": agent_name, "resource": resource, "used": used, "limit": limit})

    def budget_exhausted(self, agent_name: str, resource: str) -> None:
        self.emit(TEL_EVENT_BUDGET_EXHAUSTED, {"agent_name": agent_name, "resource": resource})

    def clear(self) -> None:
        self._events.clear()
        self._dropped_events = 0
