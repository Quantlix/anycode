"""Structured log events for agent lifecycle transitions."""

from __future__ import annotations

import time
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


class TelemetryEvent:
    """Represents a structured lifecycle event."""

    def __init__(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.name = name
        self.attributes = attributes or {}
        self.timestamp = time.monotonic()

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "attributes": self.attributes, "timestamp": self.timestamp}


class EventEmitter:
    """Collects and dispatches structured lifecycle events."""

    def __init__(self, enabled: bool = False) -> None:
        self._enabled = enabled
        self._events: list[TelemetryEvent] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def events(self) -> list[TelemetryEvent]:
        return list(self._events)

    def emit(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        if not self._enabled:
            return
        event = TelemetryEvent(name, attributes)
        self._events.append(event)

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
