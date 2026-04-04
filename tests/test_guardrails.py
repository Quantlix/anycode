"""Tests for the guardrails module: budget tracking, validators, and hooks."""

from __future__ import annotations

import pytest

from anycode.guardrails.budget import BudgetTracker, estimate_cost
from anycode.guardrails.hooks import HookRunner, LoggingHook
from anycode.guardrails.validators import (
    MAX_VALIDATION_RETRIES,
    BlocklistValidator,
    ContainsValidator,
    MaxLengthValidator,
    run_validators,
)
from anycode.types import (
    AgentInfo,
    BudgetStatus,
    GuardrailConfig,
    LLMMessage,
    LLMResponse,
    TextBlock,
    TokenUsage,
)

# -- Budget tests --


class TestBudgetTracker:
    def _make_agent_info(self) -> AgentInfo:
        return AgentInfo(name="test-agent", role="assistant", model="claude-sonnet-4-6")

    def test_no_config_never_exhausted(self) -> None:
        tracker = BudgetTracker()
        tracker.record_usage(TokenUsage(input_tokens=100000, output_tokens=100000))
        assert tracker.is_exhausted() is False

    def test_token_budget_exhaustion(self) -> None:
        config = GuardrailConfig(max_tokens_per_agent=100)
        tracker = BudgetTracker(config, model="claude-sonnet-4-6")
        assert tracker.is_exhausted() is False
        tracker.record_usage(TokenUsage(input_tokens=60, output_tokens=50))
        assert tracker.is_exhausted() is True
        assert tracker.tokens_used == 110

    def test_cost_budget_exhaustion(self) -> None:
        config = GuardrailConfig(max_cost_usd=0.001)
        tracker = BudgetTracker(config, model="claude-sonnet-4-6")
        # Claude Sonnet: $3/1M input + $15/1M output
        tracker.record_usage(TokenUsage(input_tokens=1000, output_tokens=1000))
        # Cost: (1000 * 3 + 1000 * 15) / 1_000_000 = 0.018
        assert tracker.is_exhausted() is True

    def test_turn_limit(self) -> None:
        config = GuardrailConfig(max_turns=3)
        tracker = BudgetTracker(config)
        tracker.record_turn()
        tracker.record_turn()
        assert tracker.is_exhausted() is False
        tracker.record_turn()
        assert tracker.is_exhausted() is True

    def test_tool_call_limit(self) -> None:
        config = GuardrailConfig(max_tool_calls=5)
        tracker = BudgetTracker(config)
        tracker.record_tool_call(3)
        assert tracker.is_exhausted() is False
        tracker.record_tool_call(3)
        assert tracker.is_exhausted() is True
        assert tracker.tool_calls_used == 6

    def test_blocked_tool(self) -> None:
        config = GuardrailConfig(blocked_tools=["bash", "file_write"])
        tracker = BudgetTracker(config)
        assert tracker.is_tool_blocked("bash") is True
        assert tracker.is_tool_blocked("file_write") is True
        assert tracker.is_tool_blocked("file_read") is False

    def test_no_blocked_tools_when_no_config(self) -> None:
        tracker = BudgetTracker()
        assert tracker.is_tool_blocked("bash") is False

    def test_get_status(self) -> None:
        config = GuardrailConfig(max_tokens_per_agent=1000, max_cost_usd=1.0, max_turns=10)
        tracker = BudgetTracker(config, model="gpt-4o")
        tracker.record_usage(TokenUsage(input_tokens=100, output_tokens=50))
        tracker.record_turn()
        status = tracker.get_status()
        assert isinstance(status, BudgetStatus)
        assert status.tokens_used == 150
        assert status.tokens_limit == 1000
        assert status.turns_used == 1
        assert status.turns_limit == 10
        assert status.exhausted is False

    def test_get_exhaustion_reason_tokens(self) -> None:
        config = GuardrailConfig(max_tokens_per_agent=50)
        tracker = BudgetTracker(config)
        tracker.record_usage(TokenUsage(input_tokens=30, output_tokens=30))
        reason = tracker.get_exhaustion_reason()
        assert reason is not None
        assert "Token budget exhausted" in reason

    def test_get_exhaustion_reason_cost(self) -> None:
        config = GuardrailConfig(max_cost_usd=0.0001)
        tracker = BudgetTracker(config, model="claude-sonnet-4-6")
        tracker.record_usage(TokenUsage(input_tokens=1000, output_tokens=1000))
        reason = tracker.get_exhaustion_reason()
        assert reason is not None
        assert "Cost budget exhausted" in reason

    def test_get_exhaustion_reason_turns(self) -> None:
        config = GuardrailConfig(max_turns=2)
        tracker = BudgetTracker(config)
        tracker.record_turn()
        tracker.record_turn()
        reason = tracker.get_exhaustion_reason()
        assert reason is not None
        assert "Turn limit reached" in reason

    def test_get_exhaustion_reason_tool_calls(self) -> None:
        config = GuardrailConfig(max_tool_calls=1)
        tracker = BudgetTracker(config)
        tracker.record_tool_call(2)
        reason = tracker.get_exhaustion_reason()
        assert reason is not None
        assert "Tool call limit reached" in reason

    def test_get_exhaustion_reason_none(self) -> None:
        config = GuardrailConfig(max_turns=10)
        tracker = BudgetTracker(config)
        tracker.record_turn()
        assert tracker.get_exhaustion_reason() is None

    def test_budget_status_queryable(self) -> None:
        config = GuardrailConfig(max_tokens_per_agent=500, max_turns=5)
        tracker = BudgetTracker(config)
        tracker.record_usage(TokenUsage(input_tokens=100, output_tokens=100))
        tracker.record_turn()
        tracker.record_tool_call(2)
        status = tracker.get_status()
        assert status.tokens_used == 200
        assert status.turns_used == 1
        assert status.tool_calls_used == 2
        assert status.exhausted is False


class TestEstimateCost:
    def test_known_model(self) -> None:
        cost = estimate_cost("claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == 3.0 + 15.0  # $3 input + $15 output per 1M

    def test_unknown_model_uses_fallback(self) -> None:
        cost = estimate_cost("unknown-model-xyz", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == 5.0 + 15.0  # Fallback: $5 input + $15 output


# -- Validator tests --


class TestValidators:
    def _agent_info(self) -> AgentInfo:
        return AgentInfo(name="test", role="assistant", model="test-model")

    @pytest.mark.asyncio
    async def test_max_length_valid(self) -> None:
        v = MaxLengthValidator(100)
        result = await v.validate("short text", self._agent_info())
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_max_length_invalid(self) -> None:
        v = MaxLengthValidator(5)
        result = await v.validate("too long text", self._agent_info())
        assert result.valid is False
        assert result.retry is True
        assert "maximum length" in (result.reason or "")

    @pytest.mark.asyncio
    async def test_contains_valid(self) -> None:
        v = ContainsValidator("hello")
        result = await v.validate("say hello world", self._agent_info())
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_contains_invalid(self) -> None:
        v = ContainsValidator("hello")
        result = await v.validate("goodbye world", self._agent_info())
        assert result.valid is False
        assert result.retry is True

    @pytest.mark.asyncio
    async def test_blocklist_valid(self) -> None:
        v = BlocklistValidator(["bad", "evil"])
        result = await v.validate("this is good content", self._agent_info())
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_blocklist_invalid(self) -> None:
        v = BlocklistValidator(["bad", "evil"])
        result = await v.validate("this is BAD content", self._agent_info())
        assert result.valid is False
        assert "blocked term" in (result.reason or "")

    @pytest.mark.asyncio
    async def test_run_validators_all_pass(self) -> None:
        validators = [MaxLengthValidator(1000), ContainsValidator("ok")]
        result = await run_validators("this is ok", validators, self._agent_info())
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_run_validators_first_fails(self) -> None:
        validators = [MaxLengthValidator(3), ContainsValidator("ok")]
        result = await run_validators("this is ok", validators, self._agent_info())
        assert result.valid is False
        assert "maximum length" in (result.reason or "")

    @pytest.mark.asyncio
    async def test_run_validators_empty_list(self) -> None:
        result = await run_validators("anything", [], self._agent_info())
        assert result.valid is True

    def test_max_validation_retries_is_3(self) -> None:
        assert MAX_VALIDATION_RETRIES == 3


# -- Hook tests --


class TestHooks:
    def _agent_info(self) -> AgentInfo:
        return AgentInfo(name="test", role="assistant", model="test-model")

    def _make_response(self) -> LLMResponse:
        return LLMResponse(
            id="resp-1",
            content=[TextBlock(text="test response")],
            model="test-model",
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

    @pytest.mark.asyncio
    async def test_logging_hook_before_turn(self) -> None:
        hook = LoggingHook()
        messages = [LLMMessage(role="user", content=[TextBlock(text="hello")])]
        result = await hook.before_turn(messages, self._agent_info())
        assert result == messages
        assert len(hook.before_turn_log) == 1
        assert hook.before_turn_log[0] == (1, "test")

    @pytest.mark.asyncio
    async def test_logging_hook_after_turn(self) -> None:
        hook = LoggingHook()
        response = self._make_response()
        result = await hook.after_turn(response, self._agent_info())
        assert result == response
        assert len(hook.after_turn_log) == 1
        assert hook.after_turn_log[0] == ("test", 15)

    @pytest.mark.asyncio
    async def test_hook_runner_executes_in_order(self) -> None:
        order: list[int] = []

        class Hook1:
            async def before_turn(self, messages: list[LLMMessage], context: AgentInfo) -> list[LLMMessage]:
                order.append(1)
                return messages

            async def after_turn(self, response: LLMResponse, context: AgentInfo) -> LLMResponse:
                order.append(1)
                return response

        class Hook2:
            async def before_turn(self, messages: list[LLMMessage], context: AgentInfo) -> list[LLMMessage]:
                order.append(2)
                return messages

            async def after_turn(self, response: LLMResponse, context: AgentInfo) -> LLMResponse:
                order.append(2)
                return response

        runner = HookRunner([Hook1(), Hook2()])
        messages = [LLMMessage(role="user", content=[TextBlock(text="test")])]
        await runner.run_before_turn(messages, self._agent_info())
        assert order == [1, 2]

        order.clear()
        await runner.run_after_turn(self._make_response(), self._agent_info())
        assert order == [1, 2]

    @pytest.mark.asyncio
    async def test_before_turn_hook_can_modify_messages(self) -> None:
        class InjectSystemMsg:
            async def before_turn(self, messages: list[LLMMessage], context: AgentInfo) -> list[LLMMessage]:
                return [LLMMessage(role="user", content=[TextBlock(text="injected")])] + list(messages)

            async def after_turn(self, response: LLMResponse, context: AgentInfo) -> LLMResponse:
                return response

        runner = HookRunner([InjectSystemMsg()])
        messages = [LLMMessage(role="user", content=[TextBlock(text="original")])]
        result = await runner.run_before_turn(messages, self._agent_info())
        assert len(result) == 2
        assert result[0].content[0].text == "injected"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_after_turn_hook_can_modify_response(self) -> None:
        class ModifyResponse:
            async def before_turn(self, messages: list[LLMMessage], context: AgentInfo) -> list[LLMMessage]:
                return messages

            async def after_turn(self, response: LLMResponse, context: AgentInfo) -> LLMResponse:
                return LLMResponse(
                    id=response.id,
                    content=[TextBlock(text="modified")],
                    model=response.model,
                    stop_reason=response.stop_reason,
                    usage=response.usage,
                )

        runner = HookRunner([ModifyResponse()])
        result = await runner.run_after_turn(self._make_response(), self._agent_info())
        assert result.content[0].text == "modified"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_hook_runner_empty_hooks(self) -> None:
        runner = HookRunner()
        messages = [LLMMessage(role="user", content=[TextBlock(text="test")])]
        result_msgs = await runner.run_before_turn(messages, self._agent_info())
        assert result_msgs == messages

        response = self._make_response()
        result_resp = await runner.run_after_turn(response, self._agent_info())
        assert result_resp == response

    def test_hook_runner_add(self) -> None:
        runner = HookRunner()
        assert len(runner.hooks) == 0
        runner.add(LoggingHook())
        assert len(runner.hooks) == 1
