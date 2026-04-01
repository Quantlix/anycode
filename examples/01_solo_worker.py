# Demo 01 — Solo Worker Agent
# Execute: uv run python examples/01_solo_worker.py

import asyncio
import sys

from anycode import (
    Agent,
    AnyCode,
    OrchestratorEvent,
    ToolExecutor,
    ToolRegistry,
    register_built_in_tools,
)

# --- Section A: Quick one-shot execution via the orchestrator facade ---


async def main() -> None:
    engine = AnyCode(
        config={
            "default_model": "gpt-4o-mini",
            "on_progress": lambda ev: print(
                f'>> worker "{ev.agent}" activated' if ev.type == "agent_start" else (
                    f'<< worker "{ev.agent}" finished' if ev.type == "agent_complete" else ""
                ),
            ),
        }
    )

    print("Section A: one-shot task via engine.run_agent()\n")

    outcome = await engine.run_agent(
        config={
            "name": "scripter",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "system_prompt": (
                "You are an efficient Python script writer. "
                "Produce concise, working code without unnecessary explanation. "
                "Leverage the provided tools to create and execute files."
            ),
            "tools": ["bash", "file_read", "file_write"],
            "max_turns": 6,
        },
        prompt=(
            "Write a tiny Python script at /tmp/fibonacci.py that:\n"
            "1. Defines a function fibonacci(n: int) -> int that returns the nth Fibonacci number\n"
            "2. Includes a docstring describing the algorithm\n"
            "3. At the bottom, call fibonacci(10) and print the result\n"
            "Then run the script with: python /tmp/fibonacci.py"
        ),
    )

    if not outcome.success:
        print("Worker encountered an issue:", outcome.output)
        sys.exit(1)

    print("\nWorker response:")
    print("=" * 50)
    print(outcome.output)
    print("=" * 50)
    print(f"\nMetrics — input tokens: {outcome.token_usage.input_tokens}, "
          f"output tokens: {outcome.token_usage.output_tokens}, "
          f"tool invocations: {len(outcome.tool_calls)}")

    # --- Section B: Incremental streaming through the Agent class directly ---

    print("\n\nSection B: streaming via Agent.stream()\n")

    registry = ToolRegistry()
    register_built_in_tools(registry)
    executor = ToolExecutor(registry)

    narrator = Agent(
        config={
            "name": "narrator",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "system_prompt": "You are a concise technical explainer. Respond in two sentences max.",
            "max_turns": 2,
        },
        tool_registry=registry,
        tool_executor=executor,
    )

    sys.stdout.write("Stream output: ")

    async for chunk in narrator.stream("Explain what a closure is in Python in one sentence."):
        if chunk.type == "text" and isinstance(chunk.data, str):
            sys.stdout.write(chunk.data)
        elif chunk.type == "done":
            sys.stdout.write("\n")
        elif chunk.type == "error":
            print(f"\nStream failure: {chunk.data}")

    # --- Section C: Multi-turn dialogue through Agent.prompt() ---

    print("\nSection C: conversational turns via Agent.prompt()\n")

    mentor = Agent(
        config={
            "name": "mentor",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "system_prompt": "You are a Python tutor. Give brief, practical answers.",
            "max_turns": 2,
        },
        tool_registry=ToolRegistry(),
        tool_executor=ToolExecutor(ToolRegistry()),
    )

    first_reply = await mentor.prompt("What are list comprehensions in Python?")
    print("Reply 1:", first_reply.output[:180])

    follow_up = await mentor.prompt("Show a single example filtering even numbers from a range.")
    print("\nReply 2:", follow_up.output[:280])

    print(f"\nDialogue history length: {len(mentor.get_history())} messages")
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
