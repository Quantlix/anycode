"""Tests for agent handoff (protocol, tool, executor)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from anycode.handoff.executor import HandoffExecutor
from anycode.handoff.protocol import build_handoff_system_prompt, build_handoff_user_message, trim_context
from anycode.handoff.tool import HANDOFF_TOOL_DEF, HandoffInput, _execute_handoff
from anycode.types import AgentRunResult, Handoff, HandoffRequest, LLMMessage, TextBlock, TokenUsage, ToolUseContext

# ---------------------------------------------------------------------------
# HandoffInput & tool sentinel
# ---------------------------------------------------------------------------


class TestHandoffTool:
    def test_handoff_tool_def_exists(self) -> None:
        assert HANDOFF_TOOL_DEF.name == "handoff"
        assert "hand off" in HANDOFF_TOOL_DEF.description.lower()

    def test_handoff_input_model(self) -> None:
        inp = HandoffInput(to_agent="writer", summary="Draft the report", reason="Research complete")
        assert inp.to_agent == "writer"

    async def test_execute_returns_sentinel(self) -> None:
        inp = HandoffInput(to_agent="writer", summary="Draft the report", reason="Research complete")
        ctx = MagicMock(spec=ToolUseContext)
        result = await _execute_handoff(inp, ctx)
        assert result.data.startswith("__HANDOFF__:")
        assert "writer" in result.data
        assert "Draft the report" in result.data
        assert "Research complete" in result.data
        assert result.is_error is False

    async def test_sentinel_format_parsable(self) -> None:
        inp = HandoffInput(to_agent="coder", summary="Implement feature", reason="Design done")
        ctx = MagicMock(spec=ToolUseContext)
        result = await _execute_handoff(inp, ctx)
        parts = result.data.split(":", 3)
        assert parts[0] == "__HANDOFF__"
        assert parts[1] == "coder"
        assert parts[2] == "Implement feature"
        assert parts[3] == "Design done"


# ---------------------------------------------------------------------------
# Protocol — context trimming
# ---------------------------------------------------------------------------


class TestTrimContext:
    def test_under_limit_returns_copy(self) -> None:
        msgs = [LLMMessage(role="user", content=[TextBlock(text=f"msg {i}")]) for i in range(5)]
        result = trim_context(msgs, max_messages=10)
        assert len(result) == 5
        assert result is not msgs  # Should be a copy

    def test_over_limit_trims_oldest(self) -> None:
        msgs = [LLMMessage(role="user", content=[TextBlock(text=f"msg {i}")]) for i in range(30)]
        result = trim_context(msgs, max_messages=10)
        assert len(result) == 10
        # Should keep the last 10
        assert result[0].content[0].text == "msg 20"
        assert result[-1].content[0].text == "msg 29"

    def test_exact_limit(self) -> None:
        msgs = [LLMMessage(role="user", content=[TextBlock(text=f"msg {i}")]) for i in range(20)]
        result = trim_context(msgs, max_messages=20)
        assert len(result) == 20

    def test_empty_list(self) -> None:
        result = trim_context([], max_messages=10)
        assert result == []


# ---------------------------------------------------------------------------
# Protocol — prompt builders
# ---------------------------------------------------------------------------


class TestBuildHandoffSystemPrompt:
    def test_includes_agent_and_reason(self) -> None:
        handoff = Handoff(
            id="h1",
            from_agent="researcher",
            to_agent="writer",
            context=[],
            summary="Research is done",
            reason="Need writing expertise",
            created_at=datetime.now(UTC),
        )
        prompt = build_handoff_system_prompt(handoff)
        assert "researcher" in prompt
        assert "Need writing expertise" in prompt
        assert "Research is done" in prompt


class TestBuildHandoffUserMessage:
    def test_builds_message_with_context(self) -> None:
        handoff = Handoff(
            id="h2",
            from_agent="researcher",
            to_agent="writer",
            context=[
                LLMMessage(role="user", content=[TextBlock(text="Find papers")]),
                LLMMessage(role="assistant", content=[TextBlock(text="Found 3 papers")]),
            ],
            summary="Research is done",
            reason="Need writing expertise",
            created_at=datetime.now(UTC),
        )
        msg = build_handoff_user_message(handoff)
        assert msg.role == "user"
        text = msg.content[0].text
        assert "researcher" in text
        assert "Research is done" in text
        assert "Found 3 papers" in text

    def test_builds_message_without_context(self) -> None:
        handoff = Handoff(
            id="h3",
            from_agent="a",
            to_agent="b",
            context=[],
            summary="Done",
            reason="Handoff",
            created_at=datetime.now(UTC),
        )
        msg = build_handoff_user_message(handoff)
        assert msg.role == "user"
        assert "Done" in msg.content[0].text

    def test_truncates_long_messages(self) -> None:
        long_text = "x" * 1000
        handoff = Handoff(
            id="h4",
            from_agent="a",
            to_agent="b",
            context=[LLMMessage(role="assistant", content=[TextBlock(text=long_text)])],
            summary="Done",
            reason="Handoff",
            created_at=datetime.now(UTC),
        )
        msg = build_handoff_user_message(handoff)
        text = msg.content[0].text
        assert "..." in text
        # The truncated message should be shorter than the original
        assert len(text) < 1000 + 200  # some overhead for headers


# ---------------------------------------------------------------------------
# HandoffExecutor
# ---------------------------------------------------------------------------


class TestHandoffExecutor:
    async def test_execute_success(self) -> None:
        executor = HandoffExecutor(max_depth=3)
        request = HandoffRequest(to_agent="writer", summary="Draft report", reason="Research done")
        conversation = [LLMMessage(role="user", content=[TextBlock(text="Hello")])]

        resolver = MagicMock()
        expected_result = AgentRunResult(
            success=True,
            output="Report drafted",
            messages=[],
            token_usage=TokenUsage(input_tokens=10, output_tokens=20),
            tool_calls=[],
        )
        resolver.resolve_and_run = AsyncMock(return_value=expected_result)

        result, handoff = await executor.execute(
            request=request,
            from_agent="researcher",
            conversation=conversation,
            agent_resolver=resolver,
        )

        assert result.success is True
        assert result.output == "Report drafted"
        assert handoff.from_agent == "researcher"
        assert handoff.to_agent == "writer"
        assert handoff.summary == "Draft report"
        resolver.resolve_and_run.assert_called_once()

    async def test_depth_limit_blocks(self) -> None:
        executor = HandoffExecutor(max_depth=2)
        request = HandoffRequest(to_agent="c", summary="s", reason="r")

        result, handoff = await executor.execute(
            request=request,
            from_agent="b",
            conversation=[],
            agent_resolver=MagicMock(),
            depth=2,
        )

        assert result.success is False
        assert "depth limit" in result.output.lower()

    async def test_missing_resolve_fn(self) -> None:
        executor = HandoffExecutor()
        request = HandoffRequest(to_agent="agent_b", summary="s", reason="r")

        # Resolver without resolve_and_run attribute
        resolver = object()

        result, handoff = await executor.execute(
            request=request,
            from_agent="agent_a",
            conversation=[],
            agent_resolver=resolver,
        )

        assert result.success is False
        assert "does not support handoff" in result.output

    async def test_handoff_record_has_trimmed_context(self) -> None:
        executor = HandoffExecutor(max_depth=5)
        request = HandoffRequest(to_agent="b", summary="s", reason="r")
        long_convo = [LLMMessage(role="user", content=[TextBlock(text=f"msg {i}")]) for i in range(50)]

        resolver = MagicMock()
        resolver.resolve_and_run = AsyncMock(
            return_value=AgentRunResult(
                success=True,
                output="done",
                messages=[],
                token_usage=TokenUsage(input_tokens=0, output_tokens=0),
                tool_calls=[],
            )
        )

        _, handoff = await executor.execute(
            request=request,
            from_agent="a",
            conversation=long_convo,
            agent_resolver=resolver,
        )

        # Default HANDOFF_CONTEXT_MAX_MESSAGES = 20
        assert len(handoff.context) <= 20

    async def test_handoff_record_immutable(self) -> None:
        executor = HandoffExecutor()
        request = HandoffRequest(to_agent="b", summary="s", reason="r")

        resolver = MagicMock()
        resolver.resolve_and_run = AsyncMock(
            return_value=AgentRunResult(
                success=True,
                output="ok",
                messages=[],
                token_usage=TokenUsage(input_tokens=0, output_tokens=0),
                tool_calls=[],
            )
        )

        _, handoff = await executor.execute(
            request=request,
            from_agent="a",
            conversation=[],
            agent_resolver=resolver,
        )

        # Handoff is frozen
        with pytest.raises(Exception):
            handoff.from_agent = "x"
