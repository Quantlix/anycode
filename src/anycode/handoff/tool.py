"""Built-in handoff tool — agents call this to request a handoff."""

from __future__ import annotations

import json

from pydantic import BaseModel

from anycode.constants import HANDOFF_SENTINEL_PREFIX, HANDOFF_TOOL_NAME
from anycode.types import ToolDefinition, ToolResult, ToolUseContext


class HandoffInput(BaseModel):
    """Input for the handoff tool."""

    to_agent: str
    summary: str
    reason: str


def encode_handoff_payload(to_agent: str, summary: str, reason: str) -> str:
    """Encode a handoff payload as a JSON-prefixed sentinel string.

    The runner detects this sentinel via :data:`HANDOFF_SENTINEL_PREFIX` and decodes
    the trailing JSON object via :func:`decode_handoff_payload`. JSON is used so that
    free-form ``summary``/``reason`` text containing colons or other delimiters round-trips
    losslessly.
    """
    payload = json.dumps({"to_agent": to_agent, "summary": summary, "reason": reason})
    return f"{HANDOFF_SENTINEL_PREFIX}{payload}"


def decode_handoff_payload(data: str) -> dict[str, str] | None:
    """Decode a sentinel string produced by :func:`encode_handoff_payload`.

    Returns ``None`` if the string is not a handoff sentinel or if the JSON payload is
    malformed or missing required fields.
    """
    if not data.startswith(HANDOFF_SENTINEL_PREFIX):
        return None
    raw = data[len(HANDOFF_SENTINEL_PREFIX) :]
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict):
        return None
    required = ("to_agent", "summary", "reason")
    if not all(isinstance(decoded.get(k), str) for k in required):
        return None
    return {k: decoded[k] for k in required}


async def _execute_handoff(validated_input: HandoffInput, context: ToolUseContext) -> ToolResult:
    """Sentinel execution — returns a marker result that the runner detects as a handoff signal.

    The actual handoff is orchestrated by the runner/orchestrator, not here.
    """
    return ToolResult(
        data=encode_handoff_payload(
            to_agent=validated_input.to_agent,
            summary=validated_input.summary,
            reason=validated_input.reason,
        ),
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
