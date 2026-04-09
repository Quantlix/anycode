"""Built-in handoff tool — agents call this to request a handoff."""

from __future__ import annotations

from pydantic import BaseModel

from anycode.constants import HANDOFF_TOOL_NAME
from anycode.types import ToolDefinition, ToolResult, ToolUseContext


class HandoffInput(BaseModel):
    """Input for the handoff tool."""

    to_agent: str
    summary: str
    reason: str


async def _execute_handoff(validated_input: HandoffInput, context: ToolUseContext) -> ToolResult:
    """Sentinel execution — returns a marker result that the runner detects as a handoff signal.

    The actual handoff is orchestrated by the runner/orchestrator, not here.
    """
    return ToolResult(
        data=f"__HANDOFF__:{validated_input.to_agent}:{validated_input.summary}:{validated_input.reason}",
        is_error=False,
    )


HANDOFF_TOOL_DEF = ToolDefinition(
    name=HANDOFF_TOOL_NAME,
    description=(
        "Hand off the current task to another agent. Use this when you need a "
        "different agent's expertise, are stuck, or have completed your part and "
        "another agent should continue. Provide the target agent name, a summary "
        "of the work so far, and the reason for the handoff."
    ),
    input_model=HandoffInput,
    execute=_execute_handoff,
)
