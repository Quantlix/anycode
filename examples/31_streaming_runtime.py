# Demo 31 — Runtime Streaming
# Execute: uv run python examples/31_streaming_runtime.py
#
# Demonstrates the phase-11 provider-token streaming path in the runner:
#   1. RunnerStreamingConfig(enabled=...) toggles incremental consumption
#   2. With streaming ON, `text` events reach the consumer before the turn's
#      final `done` response is assembled (proven with timestamps)
#   3. With streaming OFF, the runner falls back to a single chat() call and
#      emits one consolidated `text` event
#   4. fallback_to_chat retries via chat() if a stream dies before any output
#
# Fully deterministic: uses FakeAdapter, so no API key is required.

import asyncio
import time

from anycode.core.runner import AgentRunner
from anycode.providers.fake import FakeAdapter, FakeResponse
from anycode.tools.executor import ToolExecutor
from anycode.tools.registry import ToolRegistry
from anycode.types import LLMMessage, RunnerOptions, RunnerStreamingConfig, TextBlock

SEPARATOR = "-" * 60

# One scripted reply, sliced into 6 chunks with a small per-chunk delay so the
# incremental arrival of text is observable in wall-clock time.
SCRIPT = "Streaming lets the runner surface tokens the moment the model emits them."


def _build_runner(*, streaming: RunnerStreamingConfig) -> AgentRunner:
    adapter = FakeAdapter(
        responses=[FakeResponse(text=SCRIPT, text_chunks=6, stream_delay_seconds=0.03)],
        model_name="fake-stream",
    )
    registry = ToolRegistry()
    options = RunnerOptions(
        model="fake-stream",
        system_prompt="You are a concise assistant.",
        max_turns=2,
        agent_name="streamer",
        streaming=streaming,
    )
    return AgentRunner(adapter, registry, ToolExecutor(registry), options)


async def _drive(runner: AgentRunner) -> tuple[list[tuple[float, str]], object]:
    """Run one turn; return (elapsed, chunk) text events plus the final result."""
    seed = [LLMMessage(role="user", content=[TextBlock(text="Explain streaming.")])]
    started = time.monotonic()
    text_events: list[tuple[float, str]] = []
    final = None
    async for event in runner.stream(seed):
        if event.type == "text" and isinstance(event.data, str):
            text_events.append(((time.monotonic() - started) * 1000, event.data))
        elif event.type == "done":
            final = event.data
    return text_events, final


async def main() -> None:
    print("=== Runtime Streaming Demo ===\n")

    # --- Section A: streaming enabled (incremental) ---
    print(SEPARATOR)
    print("Section A: RunnerStreamingConfig(enabled=True)\n")

    events, result = await _drive(_build_runner(streaming=RunnerStreamingConfig(enabled=True)))
    print(f"  text events received: {len(events)} (incremental)")
    for elapsed_ms, chunk in events:
        print(f"    +{elapsed_ms:6.1f}ms  {chunk!r}")
    reassembled = "".join(chunk for _, chunk in events)
    print(f"\n  reassembled text matches script: {reassembled == SCRIPT}")
    print(f"  final turn stop_reason: {getattr(result, 'stop_reason', None)}")

    # --- Section B: streaming disabled (single chat() event) ---
    print(f"\n{SEPARATOR}")
    print("Section B: RunnerStreamingConfig(enabled=False)\n")

    events, result = await _drive(_build_runner(streaming=RunnerStreamingConfig(enabled=False)))
    print(f"  text events received: {len(events)} (consolidated)")
    for elapsed_ms, chunk in events:
        print(f"    +{elapsed_ms:6.1f}ms  {chunk!r}")
    consolidated = "".join(chunk for _, chunk in events)
    print(f"\n  consolidated text matches script: {consolidated == SCRIPT}")
    print("  note: identical final output either way - streaming is transparent")

    # --- Section C: fallback_to_chat behaviour ---
    print(f"\n{SEPARATOR}")
    print("Section C: fallback_to_chat\n")

    cfg = RunnerStreamingConfig(enabled=True, fallback_to_chat=True)
    print(f"  config: enabled={cfg.enabled}, fallback_to_chat={cfg.fallback_to_chat}")
    print("  behaviour: if a provider stream fails BEFORE any text/tool event,")
    print("  the runner retries the turn once via chat() instead of erroring.")
    print("  Once output has been emitted, a mid-stream failure propagates (no")
    print("  double-emission, no duplicated tool side effects).")

    print(f"\n{SEPARATOR}")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
