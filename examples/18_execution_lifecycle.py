# Demo 18 — Execution Lifecycle & Stop Reasons
# Execute: uv run python examples/18_execution_lifecycle.py
"""
Demonstrates the explicit execution lifecycle:
  - Strict phase state machine (initialized -> executing -> observing -> completed/failed)
  - Structured StopReason on every run (success, max_turns, budget_exceeded, doom_loop, ...)
  - LifecycleEvent stream that downstream observability tools can consume
  - Doom-loop fingerprinting via repeated tool-call detection

Real LLM calls are made when ANTHROPIC_API_KEY or OPENAI_API_KEY is present.
Conversation messages and lifecycle events are persisted to ./artifacts/lifecycle/
so you can inspect runs across executions.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from anycode import (
    Agent,
    LifecycleEmitter,
    LifecycleEvent,
    LoopDetector,
    StopReason,
    ToolExecutor,
    ToolRegistry,
    fingerprint_call,
    is_valid_transition,
    register_built_in_tools,
    stop_reasons,
)

load_dotenv()

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "lifecycle"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_provider() -> tuple[str, str]:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", "claude-haiku-4-5"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", "gpt-4o-mini"
    print("ERROR: Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")
    sys.exit(1)


PROVIDER, MODEL = _resolve_provider()


def _persist(name: str, payload: dict[str, object]) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = ARTIFACTS_DIR / f"{timestamp}_{name}.json"
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return target


# --- Section A: Pure state-machine demo (no LLM) -----------------------------


def demo_state_machine() -> list[LifecycleEvent]:
    print("\n=== A. Lifecycle state machine ===")
    seen: list[LifecycleEvent] = []
    emitter = LifecycleEmitter(
        run_id="demo-state-machine",
        agent_name="demo",
        listeners=[seen.append],
    )

    emitter.transition("executing", metadata={"turn": 1})
    emitter.transition("observing", metadata={"tool_calls": 2})
    emitter.transition("verifying")
    emitter.transition("completed", stop_reason=stop_reasons.success())

    for event in emitter.events:
        reason = event.stop_reason.code if event.stop_reason else "-"
        print(f"  phase={event.phase:<12} stop_reason={reason} metadata={event.metadata}")
    print(f"  is_terminal={emitter.is_terminal} valid(initialized->completed)={is_valid_transition('initialized', 'completed')}")
    return seen


# --- Section B: Doom-loop detection (no LLM) --------------------------------


def demo_doom_loop() -> StopReason | None:
    print("\n=== B. Doom-loop detection ===")
    detector = LoopDetector(window=4, repeat_threshold=3)
    repeated = fingerprint_call("bash", {"command": "ls -la"})
    distinct = fingerprint_call("bash", {"command": "pwd"})

    detector.record(distinct)
    detector.record(repeated)
    detector.record(repeated)
    detector.record(repeated)

    looping, pattern, repeats = detector.is_looping()
    if not looping or pattern is None:
        print("  no doom loop detected")
        return None

    reason = stop_reasons.doom_loop(pattern, repeats)
    print(f"  detected pattern={pattern[:8]}... repeats={repeats}")
    print(f"  -> stop_reason: code={reason.code} recoverable={reason.recoverable}")
    return reason


# --- Section C: Real LLM run with lifecycle capture --------------------------


async def demo_real_run() -> None:
    print(f"\n=== C. Real LLM run on {PROVIDER}:{MODEL} ===")
    registry = ToolRegistry()
    register_built_in_tools(registry)

    agent = Agent(
        config={
            "name": "lifecycle-demo",
            "model": MODEL,
            "provider": PROVIDER,
            "system_prompt": (
                "You are a concise math helper. Answer in one short sentence. Do not call tools."
            ),
            "max_turns": 3,
        },
        tool_registry=registry,
        tool_executor=ToolExecutor(registry),
    )

    result = await agent.run("What is 17 * 23? Provide only the number.")

    print(f"  success={result.success} terminal_phase={result.terminal_phase}")
    if result.stop_reason:
        print(
            "  stop_reason: "
            f"code={result.stop_reason.code} "
            f"recoverable={result.stop_reason.recoverable} "
            f"message={result.stop_reason.message!r}"
        )
    print(f"  output: {result.output[:120]}")
    print("  lifecycle phases:")
    for event in result.lifecycle_events:
        reason = event.stop_reason.code if event.stop_reason else "-"
        print(f"    - phase={event.phase:<12} stop_reason={reason} metadata={event.metadata}")

    artifact = _persist(
        "real_run",
        {
            "provider": PROVIDER,
            "model": MODEL,
            "success": result.success,
            "terminal_phase": result.terminal_phase,
            "stop_reason": result.stop_reason.model_dump() if result.stop_reason else None,
            "output": result.output,
            "tokens": result.token_usage.model_dump(),
            "lifecycle_events": [e.model_dump() for e in result.lifecycle_events],
        },
    )
    print(f"  persisted to {artifact}")


# --- Section D: Forced max_turns to verify recoverable stop ------------------


async def demo_max_turns() -> None:
    print(f"\n=== D. Max-turns stop reason on {PROVIDER}:{MODEL} ===")
    registry = ToolRegistry()
    register_built_in_tools(registry)

    agent = Agent(
        config={
            "name": "max-turns-demo",
            "model": MODEL,
            "provider": PROVIDER,
            "system_prompt": (
                "You are required to call the bash tool to run `echo hello` on every turn. "
                "Never produce a final answer without a tool call."
            ),
            "max_turns": 1,
            "tools": ["bash"],
        },
        tool_registry=registry,
        tool_executor=ToolExecutor(registry),
    )

    result = await agent.run("Run echo hello as a tool call.")

    print(f"  terminal_phase={result.terminal_phase}")
    if result.stop_reason:
        print(f"  stop_reason: code={result.stop_reason.code} recoverable={result.stop_reason.recoverable}")
    artifact = _persist(
        "max_turns_run",
        {
            "stop_reason": result.stop_reason.model_dump() if result.stop_reason else None,
            "terminal_phase": result.terminal_phase,
            "tool_calls": [tc.model_dump() for tc in result.tool_calls],
            "lifecycle_events": [e.model_dump() for e in result.lifecycle_events],
        },
    )
    print(f"  persisted to {artifact}")


async def main() -> None:
    demo_state_machine()
    demo_doom_loop()
    await demo_real_run()
    await demo_max_turns()
    print(f"\nAll artifacts written under {ARTIFACTS_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
