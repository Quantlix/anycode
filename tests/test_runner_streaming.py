"""Streaming agent-loop tests: incremental emission, parity, and chat fallback."""

from __future__ import annotations

from collections.abc import AsyncIterator

from anycode.core.runner import AgentRunner
from anycode.providers.fake import FakeAdapter, FakeResponse
from anycode.tools.executor import ToolExecutor
from anycode.tools.registry import ToolRegistry, define_tool
from anycode.types import (
    LLMMessage,
    LLMResponse,
    RunnerOptions,
    RunnerStreamingConfig,
    StreamEvent,
    TextBlock,
    TokenUsage,
    ToolResult,
    ToolUseContext,
)

MESSAGES = [LLMMessage(role="user", content=[TextBlock(text="hi")])]


def _runner(adapter: object, registry: ToolRegistry | None = None, **opts: object) -> AgentRunner:
    reg = registry or ToolRegistry()
    return AgentRunner(adapter, reg, ToolExecutor(reg), RunnerOptions(model="fake", agent_name="t", max_turns=3, **opts))  # type: ignore[arg-type]


async def test_streaming_emits_text_before_done() -> None:
    adapter = FakeAdapter(responses=[FakeResponse(text="hello world", text_chunks=3)])
    runner = _runner(adapter)

    types_in_order: list[str] = []
    async for event in runner.stream(MESSAGES):
        types_in_order.append(event.type)

    first_text = types_in_order.index("text")
    done = types_in_order.index("done")
    assert first_text < done
    # Chunked streaming produced more than one text event before completion.
    assert types_in_order.count("text") >= 2


async def test_stream_and_nonstream_parity() -> None:
    script = [FakeResponse(text="answer", input_tokens=7, output_tokens=11)]

    streamed = FakeAdapter(responses=list(script))
    non_streamed = FakeAdapter(responses=list(script))

    r_stream = await _runner(streamed, streaming=RunnerStreamingConfig(enabled=True)).run(MESSAGES)
    r_chat = await _runner(non_streamed, streaming=RunnerStreamingConfig(enabled=False)).run(MESSAGES)

    assert r_stream.output == r_chat.output
    assert r_stream.token_usage == r_chat.token_usage
    assert r_stream.stop_reason is not None and r_chat.stop_reason is not None
    assert r_stream.stop_reason.code == r_chat.stop_reason.code
    assert len(r_stream.tool_calls) == len(r_chat.tool_calls)


async def test_streaming_tool_call_executes_once() -> None:
    calls = {"n": 0}

    from pydantic import BaseModel

    class _Empty(BaseModel):
        pass

    async def _count(_input: object, _ctx: ToolUseContext) -> ToolResult:
        calls["n"] += 1
        return ToolResult(data="ok")

    registry = ToolRegistry()
    registry.register(define_tool(name="counter", description="counts", input_model=_Empty, execute=_count))

    adapter = FakeAdapter(
        responses=[
            FakeResponse(tool_calls=(("counter", {}),)),
            FakeResponse(text="finished"),
        ]
    )
    result = await _runner(adapter, registry).run(MESSAGES)
    assert calls["n"] == 1
    assert result.output == "finished"


async def test_stream_failure_falls_back_to_chat() -> None:
    """An adapter that cannot stream but can chat still completes when fallback is on."""

    class _NoStreamAdapter:
        @property
        def name(self) -> str:
            return "nostream"

        async def chat(self, messages: list[LLMMessage], options: object, **kwargs: object) -> LLMResponse:
            return LLMResponse(
                id="x",
                content=[TextBlock(text="from chat")],
                model="fake",
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=1, output_tokens=1),
            )

        def stream(self, messages: list[LLMMessage], options: object) -> AsyncIterator[StreamEvent]:
            async def _gen() -> AsyncIterator[StreamEvent]:
                raise RuntimeError("stream unsupported")
                yield  # pragma: no cover - makes this an async generator

            return _gen()

    runner = _runner(_NoStreamAdapter(), streaming=RunnerStreamingConfig(enabled=True, fallback_to_chat=True))
    result = await runner.run(MESSAGES)
    assert result.output == "from chat"
