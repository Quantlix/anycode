# Demo 11 — Agent Handoff Protocol
# Execute: uv run python examples/11_agent_handoff.py
#
# Demonstrates:
#   1. Context trimming for handoff payloads
#   2. Building system/user prompts for the receiving agent
#   3. HandoffExecutor with a real LLM-backed agent resolver
#   4. Depth limiting for handoff chains
#
# Requires an API key in .env (ANTHROPIC_API_KEY or OPENAI_API_KEY).

import asyncio
import os
import sys
from datetime import UTC, datetime

from dotenv import load_dotenv

from anycode.handoff.executor import HandoffExecutor
from anycode.handoff.protocol import build_handoff_system_prompt, build_handoff_user_message, trim_context
from anycode.handoff.tool import HANDOFF_TOOL_DEF
from anycode.providers.adapter import create_adapter
from anycode.types import (
    AgentRunResult,
    Handoff,
    HandoffRequest,
    LLMChatOptions,
    LLMMessage,
    TextBlock,
)

load_dotenv()

SEPARATOR = "-" * 60


def _resolve_provider() -> tuple[str, str] | None:
    """Return (provider, model) based on available API keys, or None."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", "claude-haiku-4-5"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", "gpt-4o-mini"
    return None


async def main() -> None:
    resolved = _resolve_provider()
    if resolved is None:
        print("ERROR: Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env to run this demo.")
        sys.exit(1)

    provider, model = resolved
    print(f"=== Agent Handoff Demo (provider={provider}, model={model}) ===\n")

    # --- Section A: Context trimming ---
    print(f"{SEPARATOR}")
    print("Section A: Context Trimming")
    messages = [LLMMessage(role="user", content=[TextBlock(text=f"Message {i}")]) for i in range(30)]
    trimmed = trim_context(messages, max_messages=5)
    print(f"  original: {len(messages)} messages")
    print(f"  trimmed:  {len(trimmed)} messages (kept last 5)")

    # --- Section B: Prompt building ---
    print(f"\n{SEPARATOR}")
    print("Section B: Handoff Prompt Building")

    handoff = Handoff(
        id="demo-1",
        from_agent="researcher",
        to_agent="writer",
        context=[
            LLMMessage(role="user", content=[TextBlock(text="Find papers on AI agents")]),
            LLMMessage(role="assistant", content=[TextBlock(text="Found 3 relevant papers on multi-agent systems")]),
        ],
        summary="Research complete with 3 papers identified",
        reason="Need writing expertise to draft the report",
        created_at=datetime.now(UTC),
    )

    sys_prompt = build_handoff_system_prompt(handoff)
    print(f"  system prompt preview: {sys_prompt[:120]}...")

    user_msg = build_handoff_user_message(handoff)
    print(f"  user message preview:  {user_msg.content[0].text[:120]}...")

    # --- Section C: HandoffExecutor with real LLM ---
    print(f"\n{SEPARATOR}")
    print("Section C: HandoffExecutor (real LLM)")

    adapter = await create_adapter(provider)
    executor = HandoffExecutor(max_depth=3)

    class RealAgentResolver:
        """Resolves handoff targets by calling the LLM directly."""

        async def resolve_and_run(self, name: str, prompt: str, system_prompt_extra: str) -> AgentRunResult:
            system = f"You are '{name}', an expert agent. {system_prompt_extra}"
            chat_messages = [LLMMessage(role="user", content=[TextBlock(text=prompt)])]
            options = LLMChatOptions(model=model, system_prompt=system, max_tokens=300)
            response = await adapter.chat(chat_messages, options)

            output_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    output_text += block.text

            return AgentRunResult(
                success=True,
                output=output_text,
                messages=chat_messages,
                token_usage=response.usage,
                tool_calls=[],
            )

    resolver = RealAgentResolver()

    request = HandoffRequest(
        to_agent="writer",
        summary="Research identified 3 key papers on multi-agent collaboration",
        reason="Need writing expertise to summarize findings",
    )

    conversation = [
        LLMMessage(role="user", content=[TextBlock(text="Research AI agent collaboration patterns")]),
        LLMMessage(role="assistant", content=[TextBlock(text="I found papers on: 1) Agent orchestration, 2) Tool use, 3) Memory sharing")]),
    ]

    agent_result, handoff_record = await executor.execute(
        request=request,
        from_agent="researcher",
        conversation=conversation,
        agent_resolver=resolver,
    )

    print(f"  handoff: {handoff_record.from_agent} -> {handoff_record.to_agent}")
    print(f"  result:  success={agent_result.success}")
    print(f"  tokens:  in={agent_result.token_usage.input_tokens}, out={agent_result.token_usage.output_tokens}")
    print(f"  output:  {agent_result.output[:200]}...")

    # --- Section D: Depth limiting ---
    print(f"\n{SEPARATOR}")
    print("Section D: Depth Limit")

    result_limited, _ = await executor.execute(
        request=request,
        from_agent="agent_c",
        conversation=[],
        agent_resolver=resolver,
        depth=3,
    )
    print(f"  at depth=3 (limit=3): success={result_limited.success}")
    print(f"  message: {result_limited.output[:80]}...")

    # --- Section E: Handoff tool definition ---
    print(f"\n{SEPARATOR}")
    print("Section E: Handoff Tool Definition")
    print(f"  tool name:        {HANDOFF_TOOL_DEF.name}")
    print(f"  tool description: {HANDOFF_TOOL_DEF.description[:80]}...")

    print(f"\n{SEPARATOR}")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
