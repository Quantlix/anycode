"""Wraps AgentRunner with persistent conversation history, lifecycle state, and streaming."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from pydantic import BaseModel

from anycode.constants import AGENT_ROLE_MAX_LENGTH, TOOL_CONTEXT_ROLE_MAX_LENGTH
from anycode.core.runner import AgentRunner
from anycode.helpers.usage_tracker import EMPTY_USAGE, merge_usage
from anycode.providers.adapter import create_adapter
from anycode.telemetry.tracer import Tracer
from anycode.tools.executor import ToolExecutor
from anycode.tools.registry import ToolRegistry
from anycode.types import (
    AgentConfig,
    AgentInfo,
    AgentRunResult,
    AgentState,
    GuardrailConfig,
    LLMMessage,
    OutputValidator,
    RunnerOptions,
    RunResult,
    StreamEvent,
    StructuredAgentResult,
    TextBlock,
    ToolDefinition,
    ToolUseContext,
    TurnHook,
)


class Agent:
    """High-level agent with state management, conversation history, and streaming support."""

    def __init__(
        self,
        config: AgentConfig | dict[str, object],
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
        *,
        tracer: Tracer | None = None,
        guardrail_config: GuardrailConfig | None = None,
        hooks: list[TurnHook] | None = None,
        output_validators: list[OutputValidator] | None = None,
        output_schema: type[BaseModel] | None = None,
    ) -> None:
        typed_config = AgentConfig.model_validate(config) if isinstance(config, dict) else config
        self.name = typed_config.name
        self.config = typed_config
        self._registry = tool_registry
        self._executor = tool_executor
        self._runner: AgentRunner | None = None
        self._state = AgentState()
        self._history: list[LLMMessage] = []
        self._tracer = tracer
        self._guardrail_config = guardrail_config
        self._hooks = hooks
        self._output_validators = output_validators
        self._output_schema = output_schema

    async def _get_runner(self) -> AgentRunner:
        if self._runner is not None:
            return self._runner

        provider = self.config.provider or "anthropic"
        adapter = await create_adapter(provider)

        opts = RunnerOptions(
            model=self.config.model,
            system_prompt=self.config.system_prompt,
            max_turns=self.config.max_turns,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            allowed_tools=self.config.tools,
            agent_name=self.name,
            agent_role=(self.config.system_prompt or "assistant")[:AGENT_ROLE_MAX_LENGTH],
        )
        self._runner = AgentRunner(
            adapter,
            self._registry,
            self._executor,
            opts,
            tracer=self._tracer,
            guardrail_config=self._guardrail_config,
            hooks=self._hooks,
            output_validators=self._output_validators,
            output_schema=self._output_schema,
        )
        return self._runner

    async def run(self, prompt: str) -> AgentRunResult:
        """Execute prompt as a standalone conversation (history is ignored)."""
        messages = [LLMMessage(role="user", content=[TextBlock(text=prompt)])]
        return await self._execute_run(messages)

    async def run_structured(self, prompt: str, schema: type[BaseModel]) -> StructuredAgentResult:  # type: ignore[type-arg]
        """Execute prompt and return a validated Pydantic model instance."""
        from anycode.structured.output import parse_structured_output

        prev_schema = self._output_schema
        self._output_schema = schema
        self._runner = None

        messages = [LLMMessage(role="user", content=[TextBlock(text=prompt)])]
        result = await self._execute_run(messages)

        self._output_schema = prev_schema
        self._runner = None

        parsed = parse_structured_output(result.output, schema) if result.success else None

        return StructuredAgentResult(
            success=result.success,
            parsed=parsed,
            output=result.output,
            messages=result.messages,
            token_usage=result.token_usage,
            tool_calls=result.tool_calls,
        )

    async def prompt(self, message: str) -> AgentRunResult:
        """Continue the ongoing conversation with a new user message."""
        user_msg = LLMMessage(role="user", content=[TextBlock(text=message)])
        self._history.append(user_msg)
        result = await self._execute_run(list(self._history))
        self._history.extend(result.messages)
        return result

    async def stream(self, prompt: str) -> AsyncGenerator[StreamEvent, None]:
        """Stream a standalone conversation response as incremental events."""
        messages = [LLMMessage(role="user", content=[TextBlock(text=prompt)])]
        async for event in self._execute_stream(messages):
            yield event

    def get_state(self) -> AgentState:
        return self._state.model_copy(deep=True)

    def get_history(self) -> list[LLMMessage]:
        return list(self._history)

    def reset(self) -> None:
        self._history.clear()
        self._state = AgentState()
        self._runner = None

    def add_tool(self, tool: ToolDefinition) -> None:
        self._registry.register(tool)

    def remove_tool(self, name: str) -> None:
        self._registry.deregister(name)

    def get_tools(self) -> list[str]:
        return [t.name for t in self._registry.list()]

    async def _execute_run(self, messages: list[LLMMessage]) -> AgentRunResult:
        self._state.status = "running"
        try:
            runner = await self._get_runner()
            result = await runner.run(messages, on_message=lambda msg: self._state.messages.append(msg))
            self._state.token_usage = merge_usage(self._state.token_usage, result.token_usage)
            self._state.status = "completed"
            return AgentRunResult(
                success=True,
                output=result.output,
                messages=result.messages,
                token_usage=result.token_usage,
                tool_calls=result.tool_calls,
            )
        except Exception as e:
            self._state.status = "error"
            self._state.error = str(e)
            return AgentRunResult(success=False, output=str(e), messages=[], token_usage=EMPTY_USAGE, tool_calls=[])

    async def _execute_stream(self, messages: list[LLMMessage]) -> AsyncGenerator[StreamEvent, None]:
        self._state.status = "running"
        try:
            runner = await self._get_runner()
            async for event in runner.stream(messages):
                if event.type == "done" and isinstance(event.data, RunResult):
                    self._state.token_usage = merge_usage(self._state.token_usage, event.data.token_usage)
                    self._state.status = "completed"
                elif event.type == "error":
                    self._state.status = "error"
                    self._state.error = str(event.data)
                yield event
        except Exception as e:
            self._state.status = "error"
            self._state.error = str(e)
            yield StreamEvent(type="error", data=e)

    def build_tool_context(self) -> ToolUseContext:
        return ToolUseContext(
            agent=AgentInfo(
                name=self.name,
                role=(self.config.system_prompt or "assistant")[:TOOL_CONTEXT_ROLE_MAX_LENGTH],
                model=self.config.model,
            )
        )
