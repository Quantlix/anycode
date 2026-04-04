# Demo 04 — Hybrid Crew with Custom Tool Definitions
# Execute: uv run python examples/04_hybrid_tooling.py

import asyncio
import json
import random
import sys
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

from anycode import (
    Agent,
    AgentConfig,
    AgentInfo,
    AgentPool,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    ToolUseContext,
    define_tool,
    register_built_in_tools,
)


# --- Custom tool: weather data fetcher ---

class WeatherInput(BaseModel):
    city: str = Field(description='City name, e.g. "Tokyo"')
    units: str = Field(default="metric", description="Temperature units: metric or imperial. Defaults to metric.")


async def lookup_weather_fn(params: WeatherInput, ctx: ToolUseContext) -> ToolResult:
    try:
        endpoint = f"https://wttr.in/{params.city}?format=j1"
        req = Request(endpoint, headers={"User-Agent": "anycode-demo/1.0"})
        with urlopen(req, timeout=6) as resp:
            payload = json.loads(resp.read())

        current = (payload.get("current_condition") or [{}])[0]
        temp_val = current.get("temp_F") if params.units == "imperial" else current.get("temp_C")
        temp_unit = "°F" if params.units == "imperial" else "°C"

        return ToolResult(
            data=json.dumps({
                "city": params.city,
                "temperature": f"{temp_val or 'N/A'}{temp_unit}",
                "humidity": f"{current.get('humidity', 'N/A')}%",
                "condition": (current.get("weatherDesc") or [{}])[0].get("value", "Unknown"),
            }),
            is_error=False,
        )
    except Exception as err:
        stub_temp = random.randint(5, 35)
        return ToolResult(
            data=json.dumps({
                "city": params.city,
                "temperature": f"{stub_temp}°C",
                "humidity": f"{random.randint(30, 90)}%",
                "condition": "Partly cloudy (stub)",
                "note": f"Live lookup failed: {err}",
            }),
            is_error=False,
        )


weather_lookup_tool = define_tool(
    name="lookup_weather",
    description="Retrieve current weather conditions for a given city.",
    input_model=WeatherInput,
    execute=lookup_weather_fn,
)


# --- Custom tool: text summarizer ---

class SummarizeInput(BaseModel):
    text: str = Field(description="The full text to summarize.")
    max_words: int = Field(default=50, description="Approximate word limit for the summary.")


async def summarize_text_fn(params: SummarizeInput, ctx: ToolUseContext) -> ToolResult:
    words = params.text.split()
    truncated = " ".join(words[: params.max_words])
    summary = (truncated + "…") if len(words) > params.max_words else truncated
    return ToolResult(data=summary, is_error=False)


summarize_text_tool = define_tool(
    name="summarize_text",
    description="Condense a block of text into a shorter summary of the specified length.",
    input_model=SummarizeInput,
    execute=summarize_text_fn,
)


# --- Agent assembly helper ---

def build_agent(cfg: AgentConfig, custom_tools: list) -> Agent:
    reg = ToolRegistry()
    register_built_in_tools(reg)
    for t in custom_tools:
        reg.register(t)
    exc = ToolExecutor(reg)
    return Agent(cfg, reg, exc)


# --- Agent configs ---

collector_cfg = AgentConfig(
    name="collector",
    model="gpt-4o-mini",
    provider="openai",
    system_prompt=(
        "You are a data collection specialist.\n"
        "Use the lookup_weather tool to gather current conditions for every city you are given.\n"
        "Return the raw data as a JSON array of objects."
    ),
    tools=["lookup_weather"],
    max_turns=8,
    temperature=0,
)

reporter_cfg = AgentConfig(
    name="reporter",
    model="gpt-4o-mini",
    provider="openai",
    system_prompt=(
        "You are a weather report writer.\n"
        "Receive raw weather data and compose a brief daily weather digest.\n"
        "Use summarize_text to shorten any over-long sections.\n"
        "Keep the final report under 250 words."
    ),
    tools=["summarize_text"],
    max_turns=4,
    temperature=0.3,
)


async def main() -> None:
    collector = build_agent(collector_cfg, [weather_lookup_tool])
    reporter = build_agent(reporter_cfg, [summarize_text_tool])

    print("Hybrid crew with custom tooling")
    print(f"Custom tools: {weather_lookup_tool.name}, {summarize_text_tool.name}")
    print()

    pool = AgentPool(1)
    pool.add(collector)
    pool.add(reporter)

    # Phase 1: collector gathers weather data
    print("[Phase 1] Collector gathering weather data...")
    collection_result = await pool.run(
        "collector",
        "Use the lookup_weather tool to fetch current conditions for:\n"
        "- London\n- New York\n- Tokyo\n- Sydney\n- São Paulo\n\n"
        "Return results as a JSON array with city, temperature, humidity, and condition fields.",
    )

    if not collection_result.success:
        print("Collector failed:", collection_result.output)
        sys.exit(1)

    tool_names = ", ".join(c.tool_name for c in collection_result.tool_calls)
    print("Collection complete. Tools used:", tool_names)

    # Phase 2: reporter composes the digest
    print("\n[Phase 2] Reporter composing weather digest...")
    digest_result = await pool.run(
        "reporter",
        f"Here is the raw weather data collected by the field team:\n\n"
        f"{collection_result.output}\n\n"
        "Compose a concise daily weather digest covering:\n"
        "- A one-line summary for each city (temperature, condition)\n"
        "- Highlight the warmest and coldest cities\n"
        "- Close with a two-sentence overall outlook\n"
        "Use summarize_text if any section exceeds 60 words.",
    )

    if not digest_result.success:
        print("Reporter failed:", digest_result.output)
        sys.exit(1)

    tool_names = ", ".join(c.tool_name for c in digest_result.tool_calls)
    print("Digest ready. Tools used:", tool_names)

    # --- Final output ---
    print("\n" + "#" * 55)

    print("\nRaw collection data (first 350 chars):")
    print(collection_result.output[:350])

    print("\nWeather digest:")
    print("-" * 55)
    print(digest_result.output)
    print("-" * 55)

    combined_input = collection_result.token_usage.input_tokens + digest_result.token_usage.input_tokens
    combined_output = collection_result.token_usage.output_tokens + digest_result.token_usage.output_tokens
    print(f"\nToken totals — input: {combined_input}, output: {combined_output}")

    # --- Bonus: standalone tool invocations ---
    print("\n--- Bonus: standalone tool invocations ---\n")

    summary_result = await summarize_text_fn(
        SummarizeInput(
            text="The quick brown fox jumps over the lazy dog repeatedly in the sunny meadow while birds sing overhead",
            max_words=8,
        ),
        ToolUseContext(agent=AgentInfo(name="demo", role="demo", model="demo")),
    )
    print(f'summarize_text(8 words) = "{summary_result.data}"')

    weather_result = await lookup_weather_fn(
        WeatherInput(city="Berlin", units="metric"),
        ToolUseContext(agent=AgentInfo(name="demo", role="demo", model="demo")),
    )
    print(f"lookup_weather(Berlin) = {weather_result.data}")


if __name__ == "__main__":
    asyncio.run(main())
