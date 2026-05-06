"""Deterministic fake LLM adapter for evaluation suites and tests.

The adapter cycles through a configured sequence of responses. Each response
can declare:

* `text` — assistant text returned to the runner.
* `tool_calls` — tuple of `(tool_name, input)` pairs to invoke.
* `usage` — optional token usage estimate.

When the response list is exhausted, the adapter returns an empty assistant
message which terminates the agent loop with `success`.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterable, Iterable
from dataclasses import dataclass, field

from anycode.types import (
    ContentBlock,
    LLMChatOptions,
    LLMMessage,
    LLMResponse,
    LLMStreamOptions,
    StreamEvent,
    TextBlock,
    TokenUsage,
    ToolUseBlock,
)


@dataclass(frozen=True)
class FakeResponse:
    """One scripted reply from the FakeAdapter."""

    text: str = ""
    tool_calls: tuple[tuple[str, dict[str, object]], ...] = ()
    input_tokens: int = 5
    output_tokens: int = 5


@dataclass
class FakeAdapter:
    """Deterministic LLMAdapter that replays a scripted sequence of responses."""

    responses: list[FakeResponse] = field(default_factory=list)
    model_name: str = "fake-model"
    _cursor: int = 0

    @property
    def name(self) -> str:
        return "fake"

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> FakeAdapter:
        return cls(responses=[FakeResponse(text=t) for t in texts])

    async def chat(self, messages: list[LLMMessage], options: LLMChatOptions) -> LLMResponse:
        del messages  # unused; the script is deterministic
        if self._cursor >= len(self.responses):
            return LLMResponse(
                id=f"fake-{uuid.uuid4().hex[:8]}",
                content=[TextBlock(text="")],
                model=options.model,
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=1, output_tokens=1),
            )
        scripted = self.responses[self._cursor]
        self._cursor += 1
        blocks: list[ContentBlock] = []
        if scripted.text:
            blocks.append(TextBlock(text=scripted.text))
        for tool_name, tool_input in scripted.tool_calls:
            blocks.append(
                ToolUseBlock(
                    id=f"toolu-{uuid.uuid4().hex[:8]}",
                    name=tool_name,
                    input=dict(tool_input),
                )
            )
        if not blocks:
            blocks.append(TextBlock(text=""))
        return LLMResponse(
            id=f"fake-{uuid.uuid4().hex[:8]}",
            content=blocks,
            model=options.model,
            stop_reason="tool_use" if scripted.tool_calls else "end_turn",
            usage=TokenUsage(input_tokens=scripted.input_tokens, output_tokens=scripted.output_tokens),
        )

    def stream(self, messages: list[LLMMessage], options: LLMStreamOptions) -> AsyncIterable[StreamEvent]:
        async def _gen() -> AsyncIterable[StreamEvent]:
            response = await self.chat(messages, options)
            for block in response.content:
                if isinstance(block, TextBlock) and block.text:
                    yield StreamEvent(type="text", data=block.text)
                elif isinstance(block, ToolUseBlock):
                    yield StreamEvent(type="tool_use", data=block)

        return _gen()
