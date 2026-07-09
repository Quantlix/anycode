"""Handoff executor — orchestrates context transfer between agents."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from anycode.constants import DEFAULT_MAX_HANDOFF_DEPTH, HANDOFF_CONTEXT_MAX_MESSAGES
from anycode.handoff.protocol import build_handoff_system_prompt, build_handoff_user_message, trim_context
from anycode.helpers.usage_tracker import EMPTY_USAGE, merge_usage
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
        chain: list[Handoff] | None = None,
    ) -> tuple[AgentRunResult, Handoff]:
        """Execute a handoff from one agent to another.

        *agent_resolver* must have a callable ``resolve_and_run(name, prompt, system_prompt_extra)``
        async method (duck-typed to avoid circular imports with Agent/Orchestrator).

        When the target agent itself emits a ``handoff_request`` in its result, the executor
        follows the chain recursively up to ``max_depth``. Each successful hop is appended to
        the optional ``chain`` list (callers may use this to populate ``TeamRunResult.handoffs``
        with the full multi-hop path).

        Returns (final_result_from_terminal_agent, first_handoff_record).
        """
        if depth >= self._max_depth:
            rejected = self._build_handoff_record(request, from_agent, conversation)
            if chain is not None:
                chain.append(rejected)
            return (
                AgentRunResult(
                    success=False,
                    output=f"Handoff chain depth limit ({self._max_depth}) reached — cannot hand off to '{request.to_agent}'.",
                    messages=[],
                    token_usage=EMPTY_USAGE,
                    tool_calls=[],
                ),
                rejected,
            )

        handoff = self._build_handoff_record(request, from_agent, conversation)
        if chain is not None:
            chain.append(handoff)
        system_extra = build_handoff_system_prompt(handoff)
        user_msg = build_handoff_user_message(handoff)

        logger.info(
            "Executing handoff: '%s' → '%s' (depth=%d, reason=%s)",
            from_agent,
            request.to_agent,
            depth,
            request.reason,
        )

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

        first_block = user_msg.content[0] if user_msg.content else None
        prompt_text = first_block.text if isinstance(first_block, TextBlock) else request.summary
        result = await resolve_fn(request.to_agent, prompt_text, system_extra)

        downstream = getattr(result, "handoff_request", None)
        if downstream is not None:
            next_result, _ = await self.execute(
                downstream,
                from_agent=request.to_agent,
                conversation=result.messages,
                agent_resolver=agent_resolver,
                depth=depth + 1,
                chain=chain,
            )
            if next_result.success:
                merged_output = next_result.output
            elif result.output:
                merged_output = f"{result.output}\n[handoff-chain] {next_result.output}"
            else:
                merged_output = next_result.output
            merged = AgentRunResult(
                success=next_result.success and result.success,
                output=merged_output,
                messages=result.messages + next_result.messages,
                token_usage=merge_usage(result.token_usage, next_result.token_usage),
                tool_calls=result.tool_calls + next_result.tool_calls,
                handoff_request=None,
            )
            return merged, handoff

        return result, handoff

    def _build_handoff_record(
        self,
        request: HandoffRequest,
        from_agent: str,
        conversation: list[LLMMessage],
    ) -> Handoff:
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
