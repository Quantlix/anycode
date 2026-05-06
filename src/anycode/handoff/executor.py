"""Handoff executor — orchestrates context transfer between agents."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from anycode.constants import DEFAULT_MAX_HANDOFF_DEPTH, HANDOFF_CONTEXT_MAX_MESSAGES
from anycode.handoff.protocol import build_handoff_system_prompt, build_handoff_user_message, trim_context
from anycode.helpers.usage_tracker import EMPTY_USAGE
from anycode.helpers.uuid7 import uuid7
from anycode.types import (
    AgentRunResult,
    Handoff,
    HandoffRequest,
    LLMMessage,
    TextBlock,
)

logger = logging.getLogger(__name__)


class HandoffExecutor:
    """Executes agent handoffs with context transfer and chain depth limiting."""

    def __init__(self, max_depth: int = DEFAULT_MAX_HANDOFF_DEPTH) -> None:
        self._max_depth = max_depth

    async def execute(
        self,
        request: HandoffRequest,
        from_agent: str,
        conversation: list[LLMMessage],
        agent_resolver: object,
        *,
        depth: int = 0,
    ) -> tuple[AgentRunResult, Handoff]:
        """Execute a handoff from one agent to another.

        *agent_resolver* must have a callable ``resolve_and_run(name, prompt, system_prompt_extra)``
        async method (duck-typed to avoid circular imports with Agent/Orchestrator).

        Returns (result_from_target_agent, handoff_record).
        """
        if depth >= self._max_depth:
            return (
                AgentRunResult(
                    success=False,
                    output=f"Handoff chain depth limit ({self._max_depth}) reached — cannot hand off to '{request.to_agent}'.",
                    messages=[],
                    token_usage=EMPTY_USAGE,
                    tool_calls=[],
                ),
                self._build_handoff_record(request, from_agent, conversation),
            )

        handoff = self._build_handoff_record(request, from_agent, conversation)
        system_extra = build_handoff_system_prompt(handoff)
        user_msg = build_handoff_user_message(handoff)

        logger.info(
            "Executing handoff: '%s' → '%s' (depth=%d, reason=%s)",
            from_agent,
            request.to_agent,
            depth,
            request.reason,
        )

        # Duck-typed call — the orchestrator/team wires this
        resolve_fn = getattr(agent_resolver, "resolve_and_run", None)
        if resolve_fn is None:
            return (
                AgentRunResult(
                    success=False,
                    output=f"Agent resolver does not support handoff — cannot reach '{request.to_agent}'.",
                    messages=[],
                    token_usage=EMPTY_USAGE,
                    tool_calls=[],
                ),
                handoff,
            )

        # Extract text from user message for prompt
        first_block = user_msg.content[0] if user_msg.content else None
        prompt_text = first_block.text if isinstance(first_block, TextBlock) else request.summary
        result = await resolve_fn(request.to_agent, prompt_text, system_extra)

        return result, handoff

    def _build_handoff_record(
        self,
        request: HandoffRequest,
        from_agent: str,
        conversation: list[LLMMessage],
    ) -> Handoff:
        """Build an immutable Handoff record."""
        trimmed = trim_context(conversation, HANDOFF_CONTEXT_MAX_MESSAGES)
        return Handoff(
            id=str(uuid7()),
            from_agent=from_agent,
            to_agent=request.to_agent,
            context=trimmed,
            summary=request.summary,
            reason=request.reason,
            created_at=datetime.now(UTC),
        )
