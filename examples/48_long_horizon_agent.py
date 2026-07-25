# Demo 48 — Long-Horizon Agent: Planning, Sub-Agents, and a Workspace
# Execute: uv run python examples/48_long_horizon_agent.py
# Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment or .env.
#
# There is no separate "deep agent" type. Planning, delegation, and a confined
# workspace are opt-in keywords on the same Agent used everywhere else.

import asyncio
import shutil
from pathlib import Path

from dotenv import load_dotenv

from anycode import Agent, SubAgentSpec, tool

load_dotenv()

WORKSPACE = Path(".anycode/example-48-workspace")


@tool
def lookup_spec(component: str) -> dict:
    """Look up the (stubbed) published specification for a system component.

    Args:
        component: Component name, e.g. "cache" or "queue".
    """
    catalog = {
        "cache": {"latency_p99_ms": 3, "eviction": "LRU", "max_entry_kb": 512},
        "queue": {"latency_p99_ms": 40, "delivery": "at-least-once", "max_message_kb": 256},
    }
    return catalog.get(component.lower(), {"error": f"no spec for {component}"})


researcher = Agent(
    name="researcher",
    instructions=(
        "You produce short, factual engineering notes. Use lookup_spec for any component "
        "figure — never invent numbers. Save your work to the workspace as you go."
    ),
    tools=[lookup_spec, "file_write", "file_read", "list_files"],
    planning=True,
    subagents=[
        SubAgentSpec(
            name="reviewer",
            instructions=(
                "You are a blunt technical reviewer. Given a note, reply with at most two "
                "sentences naming concrete problems, or the single word SOLID if there are none."
            ),
        )
    ],
    workspace=WORKSPACE,
    max_turns=18,
    temperature=0,
)


async def main() -> None:
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)

    print(f"{researcher!r}")
    print(f"Workspace: {researcher.workspace}")
    print(f"Tools:     {', '.join(sorted(definition.name for definition in researcher.tools))}\n")

    result = await researcher.run(
        "Write a short comparison note on the cache and queue components: look up both specs, "
        "save the note to comparison.md in your workspace, have the reviewer critique it, and "
        "revise once based on that critique. Finish by telling me the file path and the verdict."
    )

    print("Agent output\n" + "=" * 55)
    print(result.output)
    print("=" * 55)

    print("\nPlan as the agent last wrote it:")
    for item in researcher.todos:
        marker = {"completed": "[x]", "in_progress": "[>]", "pending": "[ ]"}[item.status]
        print(f"  {marker} {item.content}")
    if not researcher.todos:
        print("  (the agent finished without writing a plan)")

    print("\nWorkspace contents:")
    for path in sorted(WORKSPACE.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(WORKSPACE)} ({path.stat().st_size} bytes)")

    delegated = [call for call in result.tool_calls if call.tool_name == "delegate"]
    print(f"\nSucceeded:     {result.success}")
    print(f"Tool calls:    {len(result.tool_calls)} ({len(delegated)} delegated to a sub-agent)")
    print(f"Tokens:        {result.token_usage.input_tokens} in / {result.token_usage.output_tokens} out")
    print("               (sub-agent usage is merged into the parent result)")


if __name__ == "__main__":
    asyncio.run(main())
