# Demo 46 — Crew Quickstart
# Execute: uv run python examples/46_crew_quickstart.py
# Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment or .env.

import asyncio

from dotenv import load_dotenv

from anycode import Agent, Crew, TaskSpec, tool

load_dotenv()


@tool
def price_lookup(ticker: str) -> dict:
    """Look up a (stubbed) closing price for a ticker symbol.

    Args:
        ticker: Upper-case ticker symbol, e.g. "ACME".
    """
    prices = {"ACME": 128.40, "GLOBEX": 71.15, "INITECH": 42.00}
    return {"ticker": ticker, "close": prices.get(ticker.upper(), 0.0)}


analyst = Agent(
    name="analyst",
    role="a market analyst",
    goal="pull the closing prices and state the plain facts",
    tools=[price_lookup],
    max_turns=6,
    temperature=0,
)

writer = Agent(
    name="writer",
    role="a financial newsletter writer",
    goal="turn raw figures into a short, readable note",
    backstory="You write for busy readers who want the number and the takeaway, nothing else.",
    tools=[],
    max_turns=3,
)


def main() -> None:
    # --- Declared tasks with an explicit dependency ---

    crew = Crew(
        agents=[analyst, writer],
        tasks=[
            TaskSpec(
                "Collect prices",
                "Look up the closing price for ACME, GLOBEX, and INITECH using the price_lookup tool.",
                agent=analyst,
                expected_output="One line per ticker: SYMBOL — closing price.",
            ),
            TaskSpec(
                "Write the note",
                "Write a three-sentence market note from the collected prices.",
                agent=writer,
                depends_on=["Collect prices"],
                expected_output="Exactly three sentences, no bullet points.",
            ),
        ],
        verbose=True,
    )

    print(f"{crew!r}\n")
    result = crew.run_sync()

    print("\nFinal note")
    print("=" * 55)
    print(result)
    print("=" * 55)
    print(f"\nSucceeded: {result.success}")
    print(f"Per-agent outputs: {sorted(result.outputs)}")
    print(f"Tokens — input: {result.usage.input_tokens}, output: {result.usage.output_tokens}")


async def autonomous() -> None:
    # --- No tasks declared: the first agent plans the work ---

    print("\n\nAutonomous mode — the crew decomposes the goal itself\n")

    async with Crew(agents=[analyst, writer], name="research-desk") as crew:
        result = await crew.run("Report which of ACME, GLOBEX, or INITECH closed highest, and why that matters.")

    print(result.output.strip()[:600])
    print(f"\nSucceeded: {result.success}, tokens: {result.usage.input_tokens + result.usage.output_tokens}")


if __name__ == "__main__":
    main()
    asyncio.run(autonomous())
