"""Structured log events for agent lifecycle transitions."""

from __future__ import annotations

import time
from typing import Any


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
        self.emit("agent.start", {"agent_name": agent_name, "model": model})

    def agent_complete(self, agent_name: str, turns: int, tokens_used: int) -> None:
        self.emit("agent.complete", {"agent_name": agent_name, "turns": turns, "tokens_used": tokens_used})

    def agent_error(self, agent_name: str, error: str) -> None:
        self.emit("agent.error", {"agent_name": agent_name, "error": error})

    def turn_start(self, agent_name: str, turn_number: int) -> None:
        self.emit("turn.start", {"agent_name": agent_name, "turn_number": turn_number})

    def turn_complete(self, agent_name: str, turn_number: int, token_input: int, token_output: int) -> None:
        self.emit(
            "turn.complete",
            {"agent_name": agent_name, "turn_number": turn_number, "token_input": token_input, "token_output": token_output},
        )

    def tool_start(self, agent_name: str, tool_name: str) -> None:
        self.emit("tool.start", {"agent_name": agent_name, "tool_name": tool_name})

    def tool_complete(self, agent_name: str, tool_name: str, duration_ms: float, is_error: bool) -> None:
        self.emit(
            "tool.complete",
            {"agent_name": agent_name, "tool_name": tool_name, "duration_ms": duration_ms, "is_error": is_error},
        )

    def llm_call_start(self, agent_name: str, model: str) -> None:
        self.emit("llm.call.start", {"agent_name": agent_name, "model": model})

    def llm_call_complete(self, agent_name: str, model: str, input_tokens: int, output_tokens: int, duration_ms: float) -> None:
        self.emit(
            "llm.call.complete",
            {"agent_name": agent_name, "model": model, "input_tokens": input_tokens, "output_tokens": output_tokens, "duration_ms": duration_ms},
        )

    def budget_warning(self, agent_name: str, resource: str, used: float, limit: float) -> None:
        self.emit("budget.warning", {"agent_name": agent_name, "resource": resource, "used": used, "limit": limit})

    def budget_exhausted(self, agent_name: str, resource: str) -> None:
        self.emit("budget.exhausted", {"agent_name": agent_name, "resource": resource})

    def clear(self) -> None:
        self._events.clear()
