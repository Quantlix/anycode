"""Durable runs: crash a long-running agent mid-task, then resume it.

A durable run persists an append-only transcript, turn checkpoints, budget
state, and its lifecycle to ``.anycode/runs/<run_id>/`` (a temp dir here).
Killing the process mid-run loses nothing: a fresh process loads the latest
checkpoint and continues — history, turn count, and cost accounting intact.

Inspect any run store from the terminal::

    anycode runs list --root <run root>
    anycode runs show <run id> --root <run root>
    anycode runs audit <run id> --root <run root>

Run::

    uv run python examples/28_durable_runs.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from pydantic import BaseModel

from anycode import DurabilityConfig, FakeAdapter, FakeResponse, FilesystemRunStore
from anycode.core.runner import AgentRunner
from anycode.tools.executor import ToolExecutor
from anycode.tools.registry import ToolRegistry
from anycode.types import LLMMessage, RunnerOptions, TextBlock, ToolDefinition, ToolResult


def build_registry() -> ToolRegistry:
    class _Empty(BaseModel):
        pass

    async def _execute(**_kwargs: object) -> ToolResult:
        return ToolResult(data="analysis chunk processed")

    registry = ToolRegistry()
    registry.register(ToolDefinition(name="process_chunk", description="Process one data chunk", input_model=_Empty, execute=_execute))
    return registry


def build_runner(adapter: FakeAdapter, store: FilesystemRunStore, resume=None) -> AgentRunner:  # type: ignore[no-untyped-def]
    registry = build_registry()
    return AgentRunner(
        adapter,
        registry,
        ToolExecutor(registry),
        RunnerOptions(model="fake-model", agent_name="durable-worker", max_turns=10),
        durability=DurabilityConfig(enabled=True, run_root=str(store.root), checkpoint_every_turns=1),
        run_store=store,
        resume_from=resume,
    )


class CrashingAdapter(FakeAdapter):
    """Simulates a process crash after two completed turns."""

    async def chat(self, messages, options):  # type: ignore[no-untyped-def]
        if self._cursor >= 2:
            raise RuntimeError("simulated power loss")
        return await super().chat(messages, options)


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="anycode-durable-"))
    store = FilesystemRunStore(root)

    # --- first process: works two turns, then "crashes" -------------------
    crashing = CrashingAdapter(
        responses=[
            FakeResponse(text="processing chunk 1", tool_calls=(("process_chunk", {"n": 1}),)),
            FakeResponse(text="processing chunk 2", tool_calls=(("process_chunk", {"n": 2}),)),
            FakeResponse(text="never reached"),
        ]
    )
    runner = build_runner(crashing, store)
    async for _event in runner.stream([LLMMessage(role="user", content=[TextBlock(text="Process all chunks.")])]):
        pass

    record = store.list_runs()[0]
    checkpoint = store.load_latest_checkpoint(record.run_id)
    assert checkpoint is not None
    print(f"crashed run:  status={record.status}, durable turns={checkpoint.turn - 1}, "
          f"cost so far=${checkpoint.budget.cost_used:.4f}")

    # --- second process: fresh store + adapter, resume from checkpoint ----
    fresh_store = FilesystemRunStore(root)
    restored = fresh_store.load_latest_checkpoint(record.run_id)
    resumed = build_runner(FakeAdapter(responses=[FakeResponse(text="all chunks done")]), fresh_store, resume=restored)
    result = await resumed.run([])

    final = fresh_store.read_record(record.run_id)
    print(f"resumed run:  status={final.status}, output={result.output!r}, total turns={result.turns}")
    print(f"transcript:   {[e.kind for e in fresh_store.read_events(record.run_id)][-6:]} (tail)")
    print(f"inspect with: anycode runs show {record.run_id} --root {root}")

    assert final is not None and final.status == "completed"
    assert result.output == "all chunks done"


if __name__ == "__main__":
    asyncio.run(main())
