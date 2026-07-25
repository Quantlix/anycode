"""Wraps AgentRunner with persistent conversation history, lifecycle state, and streaming."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterator, Sequence
from pathlib import Path

from pydantic import BaseModel

from anycode.constants import AGENT_ROLE_MAX_LENGTH, TOOL_CONTEXT_ROLE_MAX_LENGTH
from anycode.core.defaults import default_model, detect_provider, missing_model_message, missing_provider_message
from anycode.core.runner import AgentRunner
from anycode.helpers.sync_runner import iterate_async_blocking, run_coroutine_blocking
from anycode.helpers.usage_tracker import EMPTY_USAGE, merge_usage
from anycode.helpers.uuid7 import uuid7
from anycode.identity import ExecutionContext
from anycode.providers.adapter import create_adapter
from anycode.security.redaction import safe_exception_message
from anycode.structured.output import parse_structured_output
from anycode.telemetry.tracer import Tracer
from anycode.tools.built_in import register_built_in_tools
from anycode.tools.executor import ToolExecutor
from anycode.tools.function_tool import ToolSpec, resolve_tool_specs
from anycode.tools.planning import TodoItem, TodoStore, build_todo_tool
from anycode.tools.registry import ToolRegistry
from anycode.tools.subagent import SubAgentSpec, as_subagent_spec, build_delegate_tool
from anycode.types import (
    AgentConfig,
    AgentInfo,
    AgentRunResult,
    AgentState,
    ContextPolicy,
    GuardrailConfig,
    LLMMessage,
    OutputValidator,
    ProviderResilienceConfig,
    RunnerOptions,
    RunResult,
    StreamEvent,
    StructuredAgentResult,
    TextBlock,
    TokenUsage,
    ToolDefinition,
    ToolResult,
    ToolSecurityPolicy,
    ToolUseContext,
    TurnHook,
    VerificationSensorConfig,
)

# Keyword arguments that describe the agent itself; supplying any of them alongside an
# explicit config object would silently create two sources of truth.
_CONFIG_KEYWORDS = (
    "name",
    "model",
    "provider",
    "instructions",
    "system_prompt",
    "role",
    "goal",
    "backstory",
    "max_turns",
    "max_tokens",
    "temperature",
    "mcp_servers",
    "context_policy",
    "verification",
    "tool_security",
    "provider_resilience",
    "execution_context",
)


BASH_TOOL_NAME = "bash"

PLANNING_CLAUSE = (
    "Before doing anything substantial, write a plan with the write_todos tool. Keep exactly one "
    "step in_progress, mark steps completed as you finish them, and resend the whole plan on every "
    "update. Re-read the returned checklist to stay oriented."
)

DELEGATION_CLAUSE = (
    "Use the delegate tool for self-contained sub-tasks such as focused research, review, or "
    "critique. A sub-agent starts from a blank conversation, so state the task in full and include "
    "the background it needs. Do the integration and final answer yourself."
)

WORKSPACE_CLAUSE = (
    "Your workspace is {workspace}. Write intermediate notes, drafts, and artifacts to files there "
    "rather than carrying them in the conversation, and read them back when you need them. File "
    "tools are confined to that directory."
)


class AgentConfigError(ValueError):
    """Raised when agent construction arguments are contradictory or incomplete."""


def compose_instructions(role: str | None = None, goal: str | None = None, backstory: str | None = None) -> str:
    """Build a system prompt from role/goal/backstory framing. Empty parts are omitted."""
    parts: list[str] = []
    if role:
        parts.append(f"You are {role.strip().rstrip('.')}.")
    if goal:
        parts.append(f"Your goal: {goal.strip()}")
    if backstory:
        parts.append(f"Background: {backstory.strip()}")
    return "\n\n".join(parts)


def _resolve_system_prompt(
    *,
    instructions: str | None,
    system_prompt: str | None,
    role: str | None,
    goal: str | None,
    backstory: str | None,
) -> str | None:
    if instructions and system_prompt and instructions != system_prompt:
        raise AgentConfigError("Agent received different values for instructions= and system_prompt=. They are aliases — pass only one.")
    direct = instructions or system_prompt
    framing = compose_instructions(role, goal, backstory)
    if direct and framing:
        raise AgentConfigError(
            "Agent received both instructions= and role/goal/backstory framing. "
            "Pick one: instructions= for a literal prompt, or role/goal/backstory to have one composed."
        )
    return direct or framing or None


def _build_agent_config(
    *,
    name: str,
    model: str | None,
    provider: str | None,
    system_prompt: str | None,
    max_turns: int | None,
    max_tokens: int | None,
    temperature: float | None,
    mcp_servers: list[str] | None,
    context_policy: ContextPolicy | None,
    verification: tuple[VerificationSensorConfig, ...],
    tool_security: ToolSecurityPolicy | None,
    provider_resilience: ProviderResilienceConfig | None,
    execution_context: ExecutionContext | None,
) -> AgentConfig:
    resolved_provider = provider or detect_provider()
    if resolved_provider is None:
        raise AgentConfigError(f'Agent "{name}": {missing_provider_message()}')
    resolved_model = model or default_model(resolved_provider)
    if resolved_model is None:
        raise AgentConfigError(f'Agent "{name}": {missing_model_message(resolved_provider)}')

    return AgentConfig(
        name=name,
        model=resolved_model,
        provider=resolved_provider,
        system_prompt=system_prompt,
        max_turns=max_turns,
        max_tokens=max_tokens,
        temperature=temperature,
        mcp_servers=mcp_servers,
        context_policy=context_policy,
        verification=verification,
        tool_security=tool_security,
        provider_resilience=provider_resilience,
        execution_context=execution_context,
    )


def _wire_tools(
    config: AgentConfig,
    tools: Sequence[ToolSpec] | None,
    registry: ToolRegistry | None,
    executor: ToolExecutor | None,
    capability_tools: Sequence[ToolDefinition] = (),
) -> tuple[AgentConfig, ToolRegistry, ToolExecutor]:
    """Build the registry/executor pair an agent needs when the caller did not supply one.

    ``capability_tools`` are always registered and always allowed — they come from
    capabilities the caller switched on (planning, delegation), not from ``tools=``.
    """
    if registry is None:
        registry = ToolRegistry()
        if tools is None:
            register_built_in_tools(registry)
        else:
            resolved = resolve_tool_specs(tools)
            added = {definition.name for definition in resolved}
            # Names already allowed by the config stay allowed; the explicit list extends them.
            carried = resolve_tool_specs(name for name in (config.tools or []) if name not in added)
            for definition in (*carried, *resolved):
                registry.register(definition)
            config = config.model_copy(update={"tools": [definition.name for definition in (*carried, *resolved)]})
    elif tools is not None:
        resolved = resolve_tool_specs(tools)
        for definition in resolved:
            if not registry.has(definition.name):
                registry.register(definition)
        allowed = [*(config.tools or []), *(definition.name for definition in resolved)]
        config = config.model_copy(update={"tools": list(dict.fromkeys(allowed))})

    for definition in capability_tools:
        if not registry.has(definition.name):
            registry.register(definition)
    if capability_tools and config.tools is not None:
        allowed = [*config.tools, *(definition.name for definition in capability_tools)]
        config = config.model_copy(update={"tools": list(dict.fromkeys(allowed))})

    return config, registry, executor or ToolExecutor(registry)


def _resolve_workspace(workspace: str | Path) -> Path:
    path = Path(workspace).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _workspace_policy(path: Path, allow_shell: bool) -> ToolSecurityPolicy:
    root = str(path)
    return ToolSecurityPolicy(workspace_root=root, allowed_path_roots=(root,), allow_shell=allow_shell)


def _shell_is_requested(tools: Sequence[ToolSpec] | None, config_tools: list[str] | None) -> bool:
    if tools is None:
        return config_tools is None or BASH_TOOL_NAME in config_tools
    return any(spec == BASH_TOOL_NAME or getattr(spec, "name", None) == BASH_TOOL_NAME for spec in tools)


def compose_capability_prompt(base: str | None, *, planning: bool, delegation: bool, workspace: Path | None) -> str | None:
    """Append the clauses for whichever long-horizon capabilities are switched on."""
    clauses = [clause for enabled, clause in ((planning, PLANNING_CLAUSE), (delegation, DELEGATION_CLAUSE)) if enabled]
    if workspace is not None:
        clauses.append(WORKSPACE_CLAUSE.format(workspace=workspace))
    if not clauses:
        return base
    return "\n\n".join([*([base] if base else []), *clauses])


class Agent:
    """High-level agent with state management, conversation history, and streaming support."""

    def __init__(
        self,
        config: AgentConfig | dict[str, object] | None = None,
        tool_registry: ToolRegistry | None = None,
        tool_executor: ToolExecutor | None = None,
        *,
        name: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        instructions: str | None = None,
        system_prompt: str | None = None,
        role: str | None = None,
        goal: str | None = None,
        backstory: str | None = None,
        tools: Sequence[ToolSpec] | None = None,
        max_turns: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        mcp_servers: list[str] | None = None,
        context_policy: ContextPolicy | None = None,
        verification: tuple[VerificationSensorConfig, ...] = (),
        tool_security: ToolSecurityPolicy | None = None,
        provider_resilience: ProviderResilienceConfig | None = None,
        execution_context: ExecutionContext | None = None,
        planning: bool = False,
        subagents: Sequence[SubAgentSpec | dict[str, object]] = (),
        workspace: str | Path | None = None,
        tracer: Tracer | None = None,
        guardrail_config: GuardrailConfig | None = None,
        hooks: list[TurnHook] | None = None,
        output_validators: list[OutputValidator] | None = None,
        output_schema: type[BaseModel] | None = None,
    ) -> None:
        supplied = {
            "name": name,
            "model": model,
            "provider": provider,
            "instructions": instructions,
            "system_prompt": system_prompt,
            "role": role,
            "goal": goal,
            "backstory": backstory,
            "max_turns": max_turns,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "mcp_servers": mcp_servers,
            "context_policy": context_policy,
            "verification": verification or None,
            "tool_security": tool_security,
            "provider_resilience": provider_resilience,
            "execution_context": execution_context,
        }

        if config is not None:
            conflicting = [keyword for keyword in _CONFIG_KEYWORDS if supplied.get(keyword) is not None]
            if conflicting:
                raise AgentConfigError(
                    f"Agent received both a config object and {', '.join(conflicting)}=. "
                    "Put every field in the config, or drop the config and pass keywords only."
                )
            typed_config = AgentConfig.model_validate(config) if isinstance(config, dict) else config
        else:
            if not name:
                raise AgentConfigError('Agent needs a name. Pass Agent(name="researcher", ...) or an AgentConfig as the first argument.')
            typed_config = _build_agent_config(
                name=name,
                model=model,
                provider=provider,
                system_prompt=_resolve_system_prompt(
                    instructions=instructions,
                    system_prompt=system_prompt,
                    role=role,
                    goal=goal,
                    backstory=backstory,
                ),
                max_turns=max_turns,
                max_tokens=max_tokens,
                temperature=temperature,
                mcp_servers=mcp_servers,
                context_policy=context_policy,
                verification=verification,
                tool_security=tool_security,
                provider_resilience=provider_resilience,
                execution_context=execution_context,
            )

        # Long-horizon capabilities. Each is inert unless switched on, so a plain agent's
        # prompt and tool set are untouched.
        self._todos = TodoStore()
        self._delegated_usage = EMPTY_USAGE
        subagent_specs = [as_subagent_spec(spec) for spec in subagents]
        workspace_path = _resolve_workspace(workspace) if workspace is not None else None

        if workspace_path is not None and typed_config.tool_security is None:
            allow_shell = _shell_is_requested(tools, typed_config.tools)
            typed_config = typed_config.model_copy(update={"tool_security": _workspace_policy(workspace_path, allow_shell)})

        capability_prompt = compose_capability_prompt(
            typed_config.system_prompt,
            planning=planning,
            delegation=bool(subagent_specs),
            workspace=workspace_path,
        )
        if capability_prompt != typed_config.system_prompt:
            typed_config = typed_config.model_copy(update={"system_prompt": capability_prompt})

        capability_tools: list[ToolDefinition] = []
        if planning:
            capability_tools.append(build_todo_tool(self._todos))
        if subagent_specs:
            capability_tools.append(build_delegate_tool(subagent_specs, typed_config, record_usage=self._record_delegated_usage))

        typed_config, resolved_registry, resolved_executor = _wire_tools(
            typed_config,
            tools,
            tool_registry,
            tool_executor,
            capability_tools,
        )

        self.name = typed_config.name
        self.config = typed_config
        self.workspace = workspace_path
        self._registry = resolved_registry
        self._executor = resolved_executor
        self._runner: AgentRunner | None = None
        self._state = AgentState()
        self._history: list[LLMMessage] = []
        self._tracer = tracer
        self._guardrail_config = guardrail_config
        self._hooks = hooks
        self._output_validators = output_validators
        self._output_schema = output_schema

    def _record_delegated_usage(self, usage: TokenUsage) -> None:
        self._delegated_usage = merge_usage(self._delegated_usage, usage)

    @property
    def todos(self) -> tuple[TodoItem, ...]:
        """The plan as the agent last wrote it. Empty unless ``planning=True``."""
        return self._todos.items

    async def _get_runner(self) -> AgentRunner:
        if self._runner is not None:
            return self._runner

        provider = self.config.provider or "anthropic"
        adapter = await create_adapter(provider, resilience=self.config.provider_resilience)

        opts = RunnerOptions(
            model=self.config.model,
            system_prompt=self.config.system_prompt,
            max_turns=self.config.max_turns,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            allowed_tools=self.config.tools,
            agent_name=self.name,
            agent_role=(self.config.system_prompt or "assistant")[:AGENT_ROLE_MAX_LENGTH],
            verification=self.config.verification,
            tool_security=self.config.tool_security,
            execution_context=self.config.execution_context,
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
            context_policy=self.config.context_policy,
        )
        return self._runner

    async def run(self, prompt: str) -> AgentRunResult:
        """Execute prompt as a standalone conversation (history is ignored)."""
        messages = [LLMMessage(role="user", content=[TextBlock(text=prompt)])]
        return await self._execute_run(messages)

    async def run_structured(self, prompt: str, schema: type[BaseModel]) -> StructuredAgentResult:  # type: ignore[type-arg]
        """Execute prompt and return a validated Pydantic model instance."""
        prev_schema = self._output_schema
        self._output_schema = schema
        self._runner = None

        messages = [LLMMessage(role="user", content=[TextBlock(text=prompt)])]
        try:
            result = await self._execute_run(messages)
        finally:
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

    def run_sync(self, prompt: str) -> AgentRunResult:
        """Blocking form of :meth:`run` for scripts and notebooks."""
        return run_coroutine_blocking(self.run(prompt), sync_call="Agent.run_sync()", async_call="await agent.run(...)")

    def prompt_sync(self, message: str) -> AgentRunResult:
        """Blocking form of :meth:`prompt`."""
        return run_coroutine_blocking(self.prompt(message), sync_call="Agent.prompt_sync()", async_call="await agent.prompt(...)")

    def stream_sync(self, prompt: str) -> Iterator[StreamEvent]:
        """Blocking form of :meth:`stream`, yielding events as they arrive."""
        return iterate_async_blocking(
            lambda: self.stream(prompt),
            sync_call="Agent.stream_sync()",
            async_call="async for event in agent.stream(...)",
        )

    async def call_tool(self, name: str, /, **arguments: object) -> ToolResult:
        """Invoke one of this agent's tools directly, bypassing the LLM.

        Runs under the same validation, security policy, and idempotency rules the agent
        would apply. A side-effecting tool gets a fresh idempotency key unless the
        arguments already carry one, so a one-off call does not have to invent it.
        """
        definition = self._registry.get(name)
        key: str | None = None
        if definition is not None and definition.side_effecting and not arguments.get(definition.idempotency_key_field or ""):
            key = str(uuid7())
        return await self._executor.execute(name, dict(arguments), self.build_tool_context(), idempotency_key=key)

    def call_tool_sync(self, name: str, /, **arguments: object) -> ToolResult:
        """Blocking form of :meth:`call_tool`."""
        return run_coroutine_blocking(
            self.call_tool(name, **arguments),
            sync_call="Agent.call_tool_sync()",
            async_call="await agent.call_tool(...)",
        )

    @property
    def tools(self) -> list[ToolDefinition]:
        """Tool definitions currently registered for this agent."""
        return self._registry.list()

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, model={self.config.model!r}, provider={self.config.provider!r}, tools={len(self._registry.list())})"

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
        if self.config.tools is not None and tool.name not in self.config.tools:
            self.config = self.config.model_copy(update={"tools": [*self.config.tools, tool.name]})
        self._runner = None

    def remove_tool(self, name: str) -> None:
        self._registry.deregister(name)
        if self.config.tools is not None and name in self.config.tools:
            self.config = self.config.model_copy(update={"tools": [tool_name for tool_name in self.config.tools if tool_name != name]})
        self._runner = None

    def _has_tool_definition(self, tool: ToolDefinition) -> bool:
        return self._registry.get(tool.name) is tool

    def get_tools(self) -> list[str]:
        return [t.name for t in self._registry.list()]

    async def _execute_run(self, messages: list[LLMMessage]) -> AgentRunResult:
        self._state = self._state.model_copy(update={"status": "running"})
        self._delegated_usage = EMPTY_USAGE
        collected_messages: list[LLMMessage] = []
        try:
            runner = await self._get_runner()
            result = await runner.run(messages, on_message=lambda msg: collected_messages.append(msg))
            # Sub-agents bill through their own adapters, so their usage only reaches this
            # result if it is folded in here.
            total_usage = merge_usage(result.token_usage, self._delegated_usage)
            self._state = self._state.model_copy(
                update={
                    "status": "completed",
                    "messages": [*self._state.messages, *collected_messages],
                    "token_usage": merge_usage(self._state.token_usage, total_usage),
                }
            )
            return AgentRunResult(
                success=result.stop_reason is None or result.stop_reason.code == "success",
                output=result.output,
                messages=result.messages,
                token_usage=total_usage,
                tool_calls=result.tool_calls,
                handoff_request=result.handoff_request,
                terminal_phase=result.terminal_phase,
                stop_reason=result.stop_reason,
                lifecycle_events=result.lifecycle_events,
                context_manifests=result.context_manifests,
                verification_results=result.verification_results,
                gate_decisions=result.gate_decisions,
                retries=result.retries,
            )
        except asyncio.CancelledError:
            self._state = self._state.model_copy(update={"status": "cancelled", "error": "Run cancelled by caller."})
            raise
        except Exception as e:
            message = safe_exception_message(e)
            self._state = self._state.model_copy(update={"status": "error", "error": message})
            return AgentRunResult(success=False, output=message, messages=[], token_usage=EMPTY_USAGE, tool_calls=[])

    async def _execute_stream(self, messages: list[LLMMessage]) -> AsyncGenerator[StreamEvent, None]:
        self._state = self._state.model_copy(update={"status": "running"})
        try:
            runner = await self._get_runner()
            async for event in runner.stream(messages):
                if event.type == "done" and isinstance(event.data, RunResult):
                    self._state = self._state.model_copy(
                        update={
                            "status": "completed",
                            "token_usage": merge_usage(self._state.token_usage, event.data.token_usage),
                        }
                    )
                elif event.type == "error":
                    self._state = self._state.model_copy(update={"status": "error", "error": str(event.data)})
                yield event
        except asyncio.CancelledError:
            self._state = self._state.model_copy(update={"status": "cancelled", "error": "Run cancelled by caller."})
            raise
        except Exception as e:
            self._state = self._state.model_copy(update={"status": "error", "error": safe_exception_message(e)})
            yield StreamEvent(type="error", data=e)

    def build_tool_context(self) -> ToolUseContext:
        return ToolUseContext(
            agent=AgentInfo(
                name=self.name,
                role=(self.config.system_prompt or "assistant")[:TOOL_CONTEXT_ROLE_MAX_LENGTH],
                model=self.config.model,
            ),
            security_policy=self.config.tool_security,
        )
