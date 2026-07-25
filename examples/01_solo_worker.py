# Demo 01 — Solo Worker Agent
# Execute: uv run python examples/01_solo_worker.py
# Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment or .env.

import asyncio
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from anycode import Agent, AnyCode

load_dotenv()

SCRIPT_PATH = Path(tempfile.gettempdir(), "fibonacci.py").as_posix()

# --- Section A: one agent, blocking call, no event loop to manage ---
# Provider and model are detected from whichever API key is present.


def run_scripter() -> None:
    scripter = Agent(
        name="scripter",
        instructions=(
            "You are an efficient Python script writer. "
            "Produce concise, working code without unnecessary explanation. "
            "Use the provided tools to create and execute files."
        ),
        tools=["bash", "file_read", "file_write"],
        max_turns=6,
    )
    print(f"Section A: blocking run via Agent.run_sync() — {scripter!r}\n")

    outcome = scripter.run_sync(
        f"Write a tiny Python script at {SCRIPT_PATH} that:\n"
        "1. Defines a function fibonacci(n: int) -> int that returns the nth Fibonacci number\n"
        "2. Includes a docstring describing the algorithm\n"
        "3. At the bottom, calls fibonacci(10) and prints the result\n"
        f"Then run the script with: python {SCRIPT_PATH}"
    )

    if not outcome.success:
        print("Worker encountered an issue:", outcome.output)
        sys.exit(1)

    print("Worker response:")
    print("=" * 50)
    print(outcome.output)
    print("=" * 50)
    print(
        f"\nMetrics — input tokens: {outcome.token_usage.input_tokens}, "
        f"output tokens: {outcome.token_usage.output_tokens}, "
        f"tool invocations: {len(outcome.tool_calls)}"
    )


async def main() -> None:
    # --- Section B: incremental streaming ---

    print("\n\nSection B: streaming via Agent.stream()\n")

    narrator = Agent(
        name="narrator",
        instructions="You are a concise technical explainer. Respond in two sentences max.",
        tools=[],
        max_turns=2,
    )

    sys.stdout.write("Stream output: ")
    async for chunk in narrator.stream("Explain what a closure is in Python in one sentence."):
        if chunk.type == "text" and isinstance(chunk.data, str):
            sys.stdout.write(chunk.data)
        elif chunk.type == "done":
            sys.stdout.write("\n")
        elif chunk.type == "error":
            print(f"\nStream failure: {chunk.data}")

    # --- Section C: multi-turn dialogue ---

    print("\nSection C: conversational turns via Agent.prompt()\n")

    mentor = Agent(
        name="mentor",
        role="a Python tutor",
        goal="give brief, practical answers",
        tools=[],
        max_turns=2,
    )

    first_reply = await mentor.prompt("What are list comprehensions in Python?")
    print("Reply 1:", first_reply.output[:180])

    follow_up = await mentor.prompt("Show a single example filtering even numbers from a range.")
    print("\nReply 2:", follow_up.output[:280])

    print(f"\nDialogue history length: {len(mentor.get_history())} messages")

    # --- Section D: the engine path, for orchestration-level concerns ---

    print("\nSection D: orchestrator events via AnyCode.run_agent()\n")

    engine = AnyCode(
        config={
            "on_progress": lambda ev: print(
                f'>> worker "{ev.agent}" activated'
                if ev.type == "agent_start"
                else (f'<< worker "{ev.agent}" finished' if ev.type == "agent_complete" else ""),
            ),
        }
    )

    summary = await engine.run_agent(
        config={
            "name": "summarizer",
            "model": mentor.config.model,
            "provider": mentor.config.provider,
            "system_prompt": "Summarize in one sentence.",
            "tools": [],
            "max_turns": 2,
        },
        prompt="Summarize what a Python decorator does.",
    )
    print("Summary:", summary.output.strip())
    await engine.close()

    print("\nDone.")


if __name__ == "__main__":
    run_scripter()
    asyncio.run(main())
