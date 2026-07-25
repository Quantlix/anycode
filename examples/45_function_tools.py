# Demo 45 — Function Tools with the @tool Decorator
# Execute: uv run python examples/45_function_tools.py
# Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment or .env.

import asyncio
import sys
from datetime import UTC, datetime

from dotenv import load_dotenv

from anycode import Agent, ToolUseContext, as_tool_definition, tool

load_dotenv()

# --- A plain synchronous function becomes a tool ---


@tool
def convert_currency(amount: float, rate: float, target: str = "EUR") -> dict:
    """Convert an amount using a fixed exchange rate.

    Args:
        amount: The source amount to convert.
        rate: Units of the target currency per unit of the source currency.
        target: ISO code of the target currency.
    """
    return {"converted": round(amount * rate, 2), "currency": target}


# --- An async function, with an explicit name and description ---


@tool(name="business_days", description="Count business days elapsed since a given ISO date.")
async def days_since(iso_date: str) -> int:
    """Ignored — the explicit description above wins."""
    await asyncio.sleep(0)
    start = datetime.fromisoformat(iso_date).replace(tzinfo=UTC)
    span = (datetime.now(UTC) - start).days
    return max(span - (span // 7) * 2, 0)


# --- A tool that needs to know which agent called it ---


@tool
def whoami(ctx: ToolUseContext) -> str:
    """Report the name and model of the calling agent."""
    return f"{ctx.agent.name} running {ctx.agent.model}"


async def main() -> None:
    print("Generated schemas\n" + "=" * 55)
    for fn in (convert_currency, days_since, whoami):
        definition = as_tool_definition(fn)
        schema = definition.input_model.model_json_schema()
        print(f"\n{definition.name}: {definition.description}")
        print(f"  properties: {sorted(schema.get('properties', {}))}")
        print(f"  required:   {schema.get('required', [])}")

    # Decorated functions stay ordinary callables.
    print("\nDirect call:", convert_currency(100.0, 0.92))

    analyst = Agent(
        name="analyst",
        instructions=(
            "You are a precise financial assistant. Use the provided tools for every "
            "calculation — never do the arithmetic yourself. Answer in one short sentence."
        ),
        tools=[convert_currency, days_since, whoami],
        max_turns=5,
        temperature=0,
    )

    # Tools are runnable through the agent without involving the LLM at all.
    direct = await analyst.call_tool("business_days", iso_date="2026-01-01")
    print("Direct tool call:", direct.data)

    print("\nLive agent run\n" + "=" * 55)
    result = await analyst.run("Convert 250 units at a rate of 0.92 into EUR, then tell me who you are. Use the tools.")

    if not result.success:
        print("Agent failed:", result.output)
        sys.exit(1)

    print(result.output)
    print("\nTools invoked:", ", ".join(call.tool_name for call in result.tool_calls) or "none")
    print(f"Tokens — input: {result.token_usage.input_tokens}, output: {result.token_usage.output_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
