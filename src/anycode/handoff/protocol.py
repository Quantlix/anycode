"""Handoff prompt builders and context trimming."""

from __future__ import annotations

from anycode.constants import HANDOFF_CONTEXT_MAX_MESSAGES, HANDOFF_MESSAGE_TRUNCATE_LENGTH, HANDOFF_MESSAGE_TRUNCATE_SUFFIX_AT
from anycode.types import Handoff, LLMMessage, TextBlock


def trim_context(messages: list[LLMMessage], max_messages: int = HANDOFF_CONTEXT_MAX_MESSAGES) -> list[LLMMessage]:
    """Trim conversation context to the last *max_messages* messages."""
    if len(messages) <= max_messages:
        return list(messages)
    return list(messages[-max_messages:])


def build_handoff_system_prompt(handoff: Handoff) -> str:
    """Build a system prompt addition explaining the handoff to the receiving agent."""
    return (
        f"You are receiving a handoff from agent '{handoff.from_agent}'.\n"
        f"Reason: {handoff.reason}\n"
        f"Summary: {handoff.summary}\n\n"
        "Continue the work based on the context provided below. "
        "The previous agent's conversation history is included for reference."
    )


def build_handoff_user_message(handoff: Handoff) -> LLMMessage:
    """Build a user message containing the handoff summary and key context."""
    context_parts: list[str] = [
        f"## Handoff from '{handoff.from_agent}'",
        f"**Reason:** {handoff.reason}",
        f"**Summary:** {handoff.summary}",
    ]

    if handoff.context:
        context_parts.append("\n## Previous conversation context:")
        for msg in handoff.context:
            role_label = "Assistant" if msg.role == "assistant" else "User"
            text_parts = [b.text for b in msg.content if isinstance(b, TextBlock)]
            if text_parts:
                combined = "\n".join(text_parts)
                # Truncate very long messages in context
                if len(combined) > HANDOFF_MESSAGE_TRUNCATE_LENGTH:
                    combined = combined[:HANDOFF_MESSAGE_TRUNCATE_SUFFIX_AT] + "..."
                context_parts.append(f"\n**{role_label}:** {combined}")

    return LLMMessage(
        role="user",
        content=[TextBlock(text="\n".join(context_parts))],
    )
