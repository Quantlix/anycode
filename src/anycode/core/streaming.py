"""Streaming turn assembly for the agent runner.

The runner drives a provider stream and, event by event, emits incremental
``text``/``thinking`` events to its own consumer while accumulating the final
``LLMResponse``. Adapters already emit a terminal ``done`` event carrying the
fully assembled response (handling provider-specific quirks); this module reuses
it when present and reconstructs an equivalent response otherwise so the rest of
the turn logic is identical to the non-streaming path.
"""

from __future__ import annotations

from anycode.constants import STOP_REASON_END_TURN, STOP_REASON_TOOL_USE
from anycode.helpers.usage_tracker import EMPTY_USAGE
from anycode.types import (
    ContentBlock,
    LLMResponse,
    StreamEvent,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)


class StreamStartupError(Exception):
    """Raised when a provider stream fails before any output/tool event.

    Signals that a ``chat()`` fallback is safe because nothing has been emitted
    to the consumer yet.
    """


class StreamAccumulator:
    """Collects provider stream events into a final :class:`LLMResponse`."""

    def __init__(self) -> None:
        self._text_parts: list[str] = []
        self._thinking_parts: list[str] = []
        self._tool_uses: list[ToolUseBlock] = []
        self._final: LLMResponse | None = None
        self._emitted = False
        self._event_count = 0

    @property
    def emitted_output(self) -> bool:
        """True once any user-visible text/thinking/tool event has been observed."""
        return self._emitted

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def final_response(self) -> LLMResponse | None:
        return self._final

    def observe(self, event: StreamEvent) -> None:
        self._event_count += 1
        if event.type == "text":
            self._text_parts.append(str(event.data))
            self._emitted = True
        elif event.type == "thinking":
            self._thinking_parts.append(str(event.data))
            self._emitted = True
        elif event.type == "tool_use":
            if isinstance(event.data, ToolUseBlock):
                self._tool_uses.append(event.data)
            self._emitted = True
        elif event.type == "done":
            if isinstance(event.data, LLMResponse):
                self._final = event.data
        elif event.type == "error":
            raise StreamStartupError(str(event.data))

    def build_response(self, model: str) -> LLMResponse:
        """Return the terminal response, reconstructing one if no ``done`` arrived."""
        if self._final is not None:
            return self._final

        content: list[ContentBlock] = []
        if self._thinking_parts:
            content.append(ThinkingBlock(thinking="".join(self._thinking_parts)))
        text = "".join(self._text_parts)
        if text:
            content.append(TextBlock(text=text))
        content.extend(self._tool_uses)

        return LLMResponse(
            id="",
            content=content,
            model=model,
            stop_reason=STOP_REASON_TOOL_USE if self._tool_uses else STOP_REASON_END_TURN,
            usage=EMPTY_USAGE,
        )
