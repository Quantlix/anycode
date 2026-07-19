---
title: "Let AnyCode Agents Hand Off Work to Each Other"
description: "Enable dynamic agent-to-agent handoff in AnyCode with the handoff tool and HandoffExecutor, control chain depth, and understand what context transfers."
keywords: anycode handoff, agent handoff, HandoffExecutor, HANDOFF_TOOL_DEF, HandoffRequest, agent delegation, handoff depth, multi-agent coordination
---

# Agent Handoff

Static task graphs decide the flow up front. Handoff lets an agent decide *at runtime* that a teammate is better suited — a researcher stuck on writing hands off to a writer, a triage agent escalates to a specialist. This guide shows how to enable handoff, cap the chain, and understand exactly what context moves across.

## Handoff vs. routing vs. dependencies

| Mechanism | Who decides | When |
| --- | --- | --- |
| Task `depends_on` | You, ahead of time | Fixed pipelines |
| [Routing](routing.md) | A classifier | Per task, before it runs |
| **Handoff** | The agent itself | Mid-run, when it hits its limits |

## Enable the handoff tool

An agent can hand off only if `"handoff"` is in its `tools` allowlist. Give it to any agent you want to be able to delegate. Targets are not pre-declared — the model names a teammate by `name` when it calls the tool, and the orchestrator validates that the target is a member of the team.

```python title="handoff_team.py"
from anycode import AgentConfig, TeamConfig

team_config = TeamConfig(
    name="support",
    agents=[
        AgentConfig(
            name="triage",
            provider="anthropic",
            model="claude-haiku-4-5",
            system_prompt="Triage the request. Hand off to 'billing' or 'engineer' when specialized help is needed.",
            tools=["handoff"],
        ),
        AgentConfig(name="billing", provider="anthropic", model="claude-haiku-4-5", tools=[]),
        AgentConfig(name="engineer", provider="anthropic", model="claude-sonnet-5", tools=["file_read", "grep"]),
    ],
)
```

When an agent calls the tool, its run ends, a `handoff` stream event fires, and the orchestrator runs the target agent — collecting the whole chain into `TeamRunResult.handoffs`.

## What transfers across a handoff

Handoff is a **fresh briefing**, not a full context copy. The receiver gets:

- A system-prompt preamble naming the sender and the handoff **reason**.
- A user message with the sender's **summary** plus recent conversation excerpts.

Only the last 20 messages carry over, only their text is included, and each is truncated to 500 characters. Tool-call blocks and images are dropped. Design the handing-off agent to write a good `summary` — that, not the raw transcript, is what the next agent works from.

## Cap the chain depth

To prevent agents from bouncing work back and forth forever, handoff chains are depth-limited. The default limit is 3.

```python title="depth.py"
from anycode import AnyCode

engine = AnyCode(config={"max_handoff_depth": 3})
```

When the limit is reached, the handoff fails cleanly with an explanatory result instead of recursing further.

## Drive handoff manually

For custom orchestration outside the engine, `HandoffExecutor` runs a handoff directly. You supply an `agent_resolver` exposing `async resolve_and_run(name, prompt, system_prompt_extra)`.

```python title="manual_handoff.py"
from anycode import HandoffExecutor
from anycode.types import HandoffRequest

executor = HandoffExecutor(max_depth=3)
request = HandoffRequest(
    to_agent="writer",
    summary="Research found three key papers on retrieval latency.",
    reason="Needs writing expertise to draft the summary.",
)
agent_result, handoff_record = await executor.execute(
    request=request,
    from_agent="researcher",
    conversation=conversation,
    agent_resolver=resolver,
)
```

!!! note "The tool emits a signal, not a transfer"
    Executing `HANDOFF_TOOL_DEF` on its own just returns an encoded sentinel string. The actual delegation happens when the runner detects that sentinel and the orchestrator (or your `HandoffExecutor`) runs the target agent. The depth limit is inclusive — calling `execute` at `depth=max_depth` fails immediately.

## The complete, runnable program

The snippets above are pieces of one program. Here is a complete `handoff.py` that drives a researcher-to-writer handoff through a real LLM, prints the handoff record and result, then shows the depth limit refusing a chain that has already gone as deep as it is allowed. It resolves a provider from whichever API key you have set, so it runs on Anthropic or OpenAI without edits.

```python title="handoff.py"
import asyncio
import os
import sys

from dotenv import load_dotenv

from anycode import (
    AgentRunResult,
    HandoffExecutor,
    LLMChatOptions,
    LLMMessage,
    TextBlock,
    create_adapter,
)
from anycode.types import HandoffRequest

load_dotenv()


def resolve_provider() -> tuple[str, str]:
    """Pick a provider and model from whichever API key is set."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", "claude-haiku-4-5"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", "gpt-4o-mini"
    sys.exit("Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your environment or .env file.")


PROVIDER, MODEL = resolve_provider()


class RealAgentResolver:
    """Resolve a handoff target by calling the LLM directly.

    The executor invokes ``resolve_and_run(name, prompt, system_prompt_extra)``
    on this object to run the receiving agent.
    """

    def __init__(self, adapter: object) -> None:
        self._adapter = adapter

    async def resolve_and_run(self, name: str, prompt: str, system_prompt_extra: str) -> AgentRunResult:
        system = f"You are '{name}', an expert agent. {system_prompt_extra}"
        messages = [LLMMessage(role="user", content=[TextBlock(text=prompt)])]
        options = LLMChatOptions(model=MODEL, system_prompt=system, max_tokens=300)
        response = await self._adapter.chat(messages, options)

        output_text = "".join(block.text for block in response.content if hasattr(block, "text"))
        return AgentRunResult(
            success=True,
            output=output_text,
            messages=messages,
            token_usage=response.usage,
            tool_calls=[],
        )


async def main() -> None:
    adapter = await create_adapter(PROVIDER)
    resolver = RealAgentResolver(adapter)
    executor = HandoffExecutor(max_depth=3)

    # The researcher has done its part; hand the findings to the writer.
    conversation = [
        LLMMessage(role="user", content=[TextBlock(text="Research AI agent collaboration patterns")]),
        LLMMessage(
            role="assistant",
            content=[TextBlock(text="Found papers on orchestration, tool use, and memory sharing")],
        ),
    ]
    request = HandoffRequest(
        to_agent="writer",
        summary="Research identified three key papers on multi-agent collaboration.",
        reason="Needs writing expertise to draft the summary.",
    )

    agent_result, handoff_record = await executor.execute(
        request=request,
        from_agent="researcher",
        conversation=conversation,
        agent_resolver=resolver,
    )

    print(f"Handoff: {handoff_record.from_agent} -> {handoff_record.to_agent}")
    print(f"Success: {agent_result.success}")
    print(
        f"Tokens — input: {agent_result.token_usage.input_tokens}, "
        f"output: {agent_result.token_usage.output_tokens}"
    )
    print(f"\nWriter output:\n{agent_result.output}\n")

    # The depth limit is inclusive: calling at depth == max_depth fails cleanly
    # instead of recursing further.
    limited_result, _ = await executor.execute(
        request=request,
        from_agent="writer",
        conversation=[],
        agent_resolver=resolver,
        depth=3,
    )
    print(f"At depth=3 (limit=3): success={limited_result.success}")
    print(f"Reason: {limited_result.output[:80]}")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it from the project root:

```bash
uv run python handoff.py
```

!!! tip "Tested copy"
    See [`examples/11_agent_handoff.py`](https://github.com/Quantlix/anycode/blob/main/examples/11_agent_handoff.py) for a CI-tested version that also walks through context trimming and prompt building.

## Next steps

- [Route tasks by complexity](routing.md) — the up-front alternative to runtime handoff.
- [Run a multi-agent team](multi-agent-team.md) — the team a handoff moves work within.
- [Build a support-triage system](../tutorials/support-triage.md) — handoff and routing in one project.
- [Public API](../reference/public-api.md) — `HandoffExecutor` and the handoff protocol helpers.
