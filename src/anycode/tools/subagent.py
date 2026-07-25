"""Scoped sub-agent delegation.

A sub-agent runs on a fresh conversation: only the task text and the context the caller
chooses to pass cross the boundary. That isolation is the point — it keeps the parent's
context small while a focused agent does the narrow work.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from anycode.helpers.usage_tracker import merge_usage
from anycode.types import AgentConfig, TokenUsage, ToolDefinition, ToolResult, ToolUseContext

if TYPE_CHECKING:
    from anycode.core.agent import Agent

DELEGATE_TOOL_NAME = "delegate"


class SubAgentSpec(BaseModel):
    """A focused agent the parent may delegate to."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    tools: tuple[Any, ...] = ()
    model: str | None = None
    provider: str | None = None
    max_turns: int | None = None


class DelegateInput(BaseModel):
    agent: str = Field(description="Which sub-agent to hand the task to.")
    task: str = Field(min_length=1, description="The complete task. The sub-agent sees none of your conversation.")
    context: str | None = Field(default=None, description="Any background the sub-agent needs to do the task.")


def as_subagent_spec(spec: SubAgentSpec | dict[str, object]) -> SubAgentSpec:
    return spec if isinstance(spec, SubAgentSpec) else SubAgentSpec.model_validate(spec)


def build_delegate_tool(
    subagents: Sequence[SubAgentSpec],
    parent: AgentConfig,
    *,
    record_usage: Callable[[TokenUsage], None] | None = None,
) -> ToolDefinition:
    """Create a ``delegate`` tool over *subagents*, inheriting defaults from *parent*.

    Sub-agents never receive the delegate tool themselves, so delegation depth is one by
    construction and a runaway fan-out is impossible.
    """
    by_name = {spec.name: spec for spec in subagents}
    if not by_name:
        raise ValueError("build_delegate_tool needs at least one sub-agent specification.")
    built: dict[str, Agent] = {}

    def _resolve(name: str) -> Agent:
        if name not in built:
            from anycode.core.agent import Agent

            spec = by_name[name]
            built[name] = Agent(
                name=spec.name,
                model=spec.model or parent.model,
                provider=spec.provider or parent.provider,
                instructions=spec.instructions,
                tools=list(spec.tools),
                max_turns=spec.max_turns or parent.max_turns,
                tool_security=parent.tool_security,
                provider_resilience=parent.provider_resilience,
                execution_context=parent.execution_context,
            )
        return built[name]

    async def execute(params: DelegateInput, _context: ToolUseContext) -> ToolResult:
        if params.agent not in by_name:
            return ToolResult(
                data=f'No sub-agent named "{params.agent}". Available: {", ".join(sorted(by_name))}.',
                is_error=True,
            )

        prompt = f"{params.task}\n\nContext:\n{params.context}" if params.context else params.task
        result = await _resolve(params.agent).run(prompt)
        if record_usage is not None:
            record_usage(result.token_usage)
        if not result.success:
            return ToolResult(data=f'Sub-agent "{params.agent}" failed: {result.output}', is_error=True)
        return ToolResult(data=result.output)

    roster = "\n".join(f"- {spec.name}: {spec.instructions.splitlines()[0]}" for spec in subagents)
    return ToolDefinition(
        name=DELEGATE_TOOL_NAME,
        description=(
            "Hand a self-contained task to a focused sub-agent and get its answer back. "
            "The sub-agent starts from a blank conversation, so state the task in full and "
            "include any background it needs.\n\nAvailable sub-agents:\n" + roster
        ),
        input_model=DelegateInput,
        execute=execute,
    )


def accumulate_usage(total: TokenUsage, addition: TokenUsage) -> TokenUsage:
    return merge_usage(total, addition)
