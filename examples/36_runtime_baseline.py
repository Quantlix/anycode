"""Emit a reproducible local-runtime baseline as JSON.

Run with::

    uv run python examples/36_runtime_baseline.py
    uv run python examples/36_runtime_baseline.py --output artifacts/runtime-baseline.json

The fixture covers task admission, deterministic execution, checkpoint size,
event volume, and context growth without provider credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from anycode import (
    AgentRunner,
    CheckpointData,
    FakeAdapter,
    FakeResponse,
    FilesystemCheckpointStore,
    LLMMessage,
    RunnerOptions,
    TaskQueue,
    TextBlock,
    TokenUsage,
    ToolExecutor,
    ToolRegistry,
    create_task,
    estimate_messages_tokens,
)

BASELINE_SCHEMA_VERSION = 1
ADMISSION_TASKS = 250
EXECUTION_RUNS = 20
CHECKPOINT_TASKS = 25
CONTEXT_MESSAGE_COUNTS = (1, 8, 32)
CONTEXT_MESSAGE_CHARACTERS = 256
MILLISECONDS_PER_SECOND = 1_000
MICROSECONDS_PER_SECOND = 1_000_000


def _elapsed_seconds(started_at: float) -> float:
    return time.perf_counter() - started_at


def _measure_admission() -> dict[str, int | float]:
    tasks = [create_task(title=f"task-{index}", description="baseline admission") for index in range(ADMISSION_TASKS)]
    queue = TaskQueue()
    started_at = time.perf_counter()
    queue.add_batch(tasks)
    elapsed = _elapsed_seconds(started_at)
    return {
        "tasks": len(queue.list()),
        "elapsed_ms": round(elapsed * MILLISECONDS_PER_SECOND, 3),
        "microseconds_per_task": round(elapsed * MICROSECONDS_PER_SECOND / ADMISSION_TASKS, 3),
    }


async def _run_once() -> tuple[float, list[str], int]:
    registry = ToolRegistry()
    runner = AgentRunner(
        FakeAdapter(responses=[FakeResponse(text="baseline complete")]),
        registry,
        ToolExecutor(registry),
        RunnerOptions(model="fake-model", max_turns=1, agent_name="baseline"),
    )
    event_types: list[str] = []
    lifecycle_events = 0
    started_at = time.perf_counter()
    async for event in runner.stream([LLMMessage(role="user", content=[TextBlock(text="run")])]):
        event_types.append(event.type)
        if event.type == "done":
            lifecycle_events = len(event.data.lifecycle_events)  # type: ignore[union-attr]
    return _elapsed_seconds(started_at), event_types, lifecycle_events


async def _measure_execution_and_events() -> tuple[dict[str, int | float], dict[str, object]]:
    durations: list[float] = []
    representative_events: list[str] = []
    lifecycle_events = 0
    for _ in range(EXECUTION_RUNS):
        elapsed, event_types, lifecycle_count = await _run_once()
        durations.append(elapsed)
        representative_events = event_types
        lifecycle_events = lifecycle_count

    total = sum(durations)
    return (
        {
            "runs": EXECUTION_RUNS,
            "elapsed_ms": round(total * MILLISECONDS_PER_SECOND, 3),
            "milliseconds_per_run": round(total * MILLISECONDS_PER_SECOND / EXECUTION_RUNS, 3),
        },
        {
            "stream_events_per_run": len(representative_events),
            "stream_event_types": dict(sorted(Counter(representative_events).items())),
            "lifecycle_events_per_run": lifecycle_events,
        },
    )


async def _measure_checkpoint() -> dict[str, int]:
    tasks = [create_task(title=f"checkpoint-{index}", description="baseline checkpoint") for index in range(CHECKPOINT_TASKS)]
    checkpoint = CheckpointData(
        id="baseline-checkpoint",
        workflow_id="baseline-workflow",
        tasks=tasks,
        agent_results={},
        wave_index=0,
        total_token_usage=TokenUsage(),
        created_at=datetime.now(UTC),
    )
    with tempfile.TemporaryDirectory(prefix="anycode-baseline-") as directory:
        root = Path(directory)
        await FilesystemCheckpointStore(str(root)).save(checkpoint)
        checkpoint_bytes = sum(path.stat().st_size for path in root.rglob("*.json"))
    return {"tasks": CHECKPOINT_TASKS, "bytes": checkpoint_bytes}


def _measure_context_growth() -> dict[str, object]:
    measurements: list[dict[str, int]] = []
    payload = "x" * CONTEXT_MESSAGE_CHARACTERS
    for count in CONTEXT_MESSAGE_COUNTS:
        messages = [LLMMessage(role="user", content=[TextBlock(text=payload)]) for _ in range(count)]
        measurements.append({"messages": count, "estimated_tokens": estimate_messages_tokens(messages)})
    return {"message_characters": CONTEXT_MESSAGE_CHARACTERS, "measurements": measurements}


async def collect_baseline() -> dict[str, object]:
    execution, events = await _measure_execution_and_events()
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "environment": {"python": platform.python_version(), "platform": platform.system().lower()},
        "metrics": {
            "task_admission": _measure_admission(),
            "execution": execution,
            "checkpoint_size": await _measure_checkpoint(),
            "event_volume": events,
            "context_growth": _measure_context_growth(),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write JSON evidence to this path instead of stdout.")
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    rendered = json.dumps(await collect_baseline(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    asyncio.run(main())
