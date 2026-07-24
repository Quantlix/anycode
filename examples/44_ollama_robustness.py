# Demo 44 — Robust Ollama Integration
# Execute: uv run python examples/44_ollama_robustness.py
#
# Demonstrates the hardened Ollama adapter:
#   1. Constructor knobs: keep_alive, think, default sampling options, and
#      API keys for ollama.com cloud (local servers ignore the header)
#   2. Thinking support: reasoning_effort tiers map to Ollama think levels and
#      surface as ThinkingBlock content / `thinking` stream events
#   3. Structured outputs: OpenAI-style response_format translates to Ollama's
#      native `format` field (json_object -> "json", json_schema -> raw schema)
#   4. done_reason mapping: stop/length/tool calls map to portable stop_reasons
#   5. Robust errors: missing models produce an actionable `ollama pull` hint,
#      and mid-stream failures surface as `error` events instead of hanging
#
# Section A is offline. Live sections need a reachable Ollama server:
#   OLLAMA_BASE_URL (default http://localhost:11434)
#   OLLAMA_MODEL    (default qwen3.5:9b — any thinking+tools model works)

import asyncio
import json
import os

from dotenv import load_dotenv

from anycode.providers.ollama import OllamaAdapter
from anycode.types import LLMChatOptions, LLMMessage, LLMStreamOptions, LLMToolDef, TextBlock

load_dotenv()

SEPARATOR = "-" * 60

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")


def user_message(text: str) -> list[LLMMessage]:
    return [LLMMessage(role="user", content=[TextBlock(text=text)])]


async def server_reachable() -> bool:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{BASE_URL}/api/version")
            response.raise_for_status()
        return True
    except Exception:
        return False


async def main() -> None:
    print("=== Robust Ollama Integration Demo ===\n")

    # --- Section A (offline): the adapter's configuration surface ---
    print(SEPARATOR)
    print("Section A: constructor knobs\n")

    adapter = OllamaAdapter(
        base_url=BASE_URL,
        model=MODEL,
        keep_alive="10m",  # keep the model resident between calls
        default_options={"num_ctx": 8192, "top_p": 0.9},  # forwarded as Ollama `options`
    )
    print(f"  adapter.name = {adapter.name}")
    print(f"  base_url     = {BASE_URL}")
    print("  keep_alive   = 10m, default options = num_ctx/top_p")
    print('  cloud usage  = OllamaAdapter(base_url="https://ollama.com", api_key=...)')
    print("                 (or set OLLAMA_API_KEY; local servers ignore the header)")

    if not await server_reachable():
        print(f"\nskipped live sections: no Ollama server at {BASE_URL}")
        print("set OLLAMA_BASE_URL / start `ollama serve` to run them")
        return

    # --- Section B (live): thinking via reasoning_effort ---
    print(f"\n{SEPARATOR}")
    print(f"Section B: thinking (model={MODEL})\n")

    # Thinking models need headroom for BOTH the reasoning and the answer. A
    # tight budget can be fully consumed by thinking, truncating (stop_reason=
    # max_tokens) before any text block is emitted — so give it real room.
    options = LLMChatOptions(model=MODEL, max_tokens=4096, reasoning_effort="low")
    response = await adapter.chat(user_message("What is 17 * 24? Answer with the number only."), options)
    has_text = False
    for block in response.content:
        if block.type == "thinking":
            print(f"  thinking ({len(block.thinking)} chars): {block.thinking[:80]!r}...")
        elif block.type == "text":
            has_text = True
            print(f"  text: {block.text.strip()!r}")
    if not has_text and response.stop_reason == "max_tokens":
        print("  (no answer text: thinking consumed the whole budget — raise max_tokens)")
    print(f"  stop_reason={response.stop_reason} usage={response.usage.input_tokens}/{response.usage.output_tokens}")

    # --- Section C (live): structured outputs ---
    print(f"\n{SEPARATOR}")
    print("Section C: structured outputs\n")

    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "country": {"type": "string"},
            "population_millions": {"type": "number"},
        },
        "required": ["city", "country", "population_millions"],
    }
    # Disable thinking for constrained decoding: a thinking model would spend
    # the token budget reasoning before emitting the JSON payload.
    json_adapter = OllamaAdapter(base_url=BASE_URL, model=MODEL, think=False)
    response = await json_adapter.chat(
        user_message("Give one fact card for Tokyo."),
        LLMChatOptions(model=MODEL, max_tokens=768),
        response_format={"type": "json_schema", "json_schema": {"schema": schema}},
    )
    raw = "".join(block.text for block in response.content if block.type == "text")
    print(f"  raw: {raw.strip()}")
    print(f"  parsed keys: {sorted(json.loads(raw))}")

    # --- Section D (live): streaming with thinking + done_reason mapping ---
    print(f"\n{SEPARATOR}")
    print("Section D: streaming events\n")

    counts = {"thinking": 0, "text": 0, "error": 0}
    final = None
    stream_options = LLMStreamOptions(model=MODEL, max_tokens=4096, reasoning_effort="low")
    async for event in adapter.stream(user_message("Name two prime numbers under 10."), stream_options):
        if event.type in counts:
            counts[event.type] += 1
        elif event.type == "done":
            final = event.data
    print(f"  events: {counts}")
    if counts["text"] == 0 and final is not None and final.stop_reason == "max_tokens":
        print("  (no text streamed: thinking consumed the whole budget — raise max_tokens)")
    if final is not None:
        print(f"  done: stop_reason={final.stop_reason} usage={final.usage.input_tokens}/{final.usage.output_tokens}")

    # --- Section E (live): tool calls map done_reason -> tool_use ---
    print(f"\n{SEPARATOR}")
    print("Section E: tool calls\n")

    weather_tool = LLMToolDef(
        name="get_weather",
        description="Get current weather for a city",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    )
    response = await adapter.chat(
        user_message("Use the get_weather tool to check the weather in Paris."),
        LLMChatOptions(model=MODEL, max_tokens=768, tools=[weather_tool]),
    )
    tool_calls = [block for block in response.content if block.type == "tool_use"]
    for call in tool_calls:
        print(f"  tool_use: {call.name}({call.input}) id={call.id[:16]}...")
    print(f"  stop_reason={response.stop_reason} (mapped from Ollama's done_reason + tool_calls)")

    # --- Section F (live): actionable missing-model errors ---
    print(f"\n{SEPARATOR}")
    print("Section F: missing-model error\n")

    ghost = OllamaAdapter(base_url=BASE_URL, model="no-such-model:1b")
    try:
        await ghost.chat(user_message("hi"), LLMChatOptions(model="no-such-model:1b", max_tokens=16))
    except Exception as error:
        print(f"  {type(error).__name__}: {error}")

    print(f"\n{SEPARATOR}\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
