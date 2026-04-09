"""Agent handoff protocol for AnyCode."""

from anycode.handoff.executor import HandoffExecutor
from anycode.handoff.protocol import build_handoff_system_prompt, build_handoff_user_message, trim_context
from anycode.handoff.tool import HANDOFF_TOOL_DEF, HandoffInput

__all__ = [
    "HandoffExecutor",
    "HandoffInput",
    "HANDOFF_TOOL_DEF",
    "build_handoff_system_prompt",
    "build_handoff_user_message",
    "trim_context",
]
