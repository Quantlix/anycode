"""Tests for the durable run store, mid-run checkpoints, and kill-and-resume."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from anycode.checkpoint.serializer import deserialize_checkpoint, serialize_checkpoint
from anycode.core.runner import AgentRunner
from anycode.providers.fake import FakeAdapter, FakeResponse
from anycode.runstore.store import FilesystemRunStore
from anycode.tools.executor import ToolExecutor
from anycode.tools.registry import ToolRegistry
from anycode.types import (
    AgentRunResult,
    CheckpointData,
    DurabilityConfig,
    LifecycleEvent,
    LLMMessage,
    QualityGateDecision,
    RunnerOptions,
    StopReason,
    TextBlock,
    TokenUsage,
    TurnCheckpoint,
    VerificationResult,
)


def _echo_tool() -> object:
    from pydantic import BaseModel as _BM

    from anycode.types import ToolDefinition, ToolResult

    class _Empty(_BM):
        pass

    async def _execute(**_kwargs: object) -> ToolResult:
        return ToolResult(data="echoed")

    return ToolDefinition(name="echo", description="echo", input_model=_Empty, execute=_execute)


def _runner(adapter: FakeAdapter, store: FilesystemRunStore, *, resume=None, every: int = 1) -> AgentRunner:  # type: ignore[no-untyped-def]
    registry = ToolRegistry()
    registry.register(_echo_tool())  # type: ignore[arg-type]
    executor = ToolExecutor(registry)
    options = RunnerOptions(model="fake-model", max_turns=10, agent_name="durable")
    return AgentRunner(
        adapter,
        registry,
        executor,
        options,
        durability=DurabilityConfig(enabled=True, run_root=str(store.root), checkpoint_every_turns=every),
        run_store=store,
        resume_from=resume,
    )


# -- store primitives --


def test_run_record_lifecycle(tmp_path: Path) -> None:
    store = FilesystemRunStore(tmp_path)
    record = store.create_run("run-1", agent_name="a", model="m")
    assert record.status == "running"
    assert store.read_record("run-1") is not None

    updated = store.update_status("run-1", "paused")
    assert updated.status == "paused"
    assert [r.run_id for r in store.list_runs()] == ["run-1"]


def test_transcript_append_and_read(tmp_path: Path) -> None:
    store = FilesystemRunStore(tmp_path)
    store.create_run("run-1", agent_name="a", model="m")
    store.append_event("run-1", "message", {"role": "assistant"})
    store.append_event("run-1", "tool_result", {"tool_name": "echo"})

    events = store.read_events("run-1")
    assert [e.seq for e in events] == [1, 2]
    assert [e.kind for e in events] == ["message", "tool_result"]
    assert store.read_events("run-1", after_seq=1)[0].kind == "tool_result"


def test_transcript_torn_tail_is_skipped(tmp_path: Path) -> None:
    store = FilesystemRunStore(tmp_path)
    store.create_run("run-1", agent_name="a", model="m")
    store.append_event("run-1", "message", {})
    transcript = tmp_path / "run-1" / "transcript.jsonl"
    with transcript.open("a", encoding="utf-8") as fh:
        fh.write('{"seq": 2, "ts": "2026-01-01T00:00:0')  # torn mid-write

    events = store.read_events("run-1")
    assert len(events) == 1

    # Appending after recovery continues the sequence from the last valid event.
    fresh_store = FilesystemRunStore(tmp_path)
    event = fresh_store.append_event("run-1", "stop", {})
    assert event.seq == 2


def test_turn_checkpoint_roundtrip_and_prune(tmp_path: Path) -> None:
    store = FilesystemRunStore(tmp_path)
    store.create_run("run-1", agent_name="a", model="m")
    for turn in range(1, 6):
        cp = TurnCheckpoint(
            run_id="run-1",
            turn=turn,
            messages=[LLMMessage(role="user", content=[TextBlock(text=f"turn {turn}")])],
            token_usage=TokenUsage(input_tokens=turn, output_tokens=turn),
            created_at=datetime.now(UTC),
        )
        store.save_checkpoint(cp, keep_last=3)

    latest = store.load_latest_checkpoint("run-1")
    assert latest is not None
    assert latest.turn == 5
    assert latest.messages[0].content[0].text == "turn 5"  # type: ignore[union-attr]
    kept = sorted((tmp_path / "run-1" / "checkpoints").glob("turn-*.json"))
    assert len(kept) == 3


def test_mark_interrupted_runs(tmp_path: Path) -> None:
    store = FilesystemRunStore(tmp_path)
    store.create_run("stale", agent_name="a", model="m")
    store.create_run("done", agent_name="a", model="m")
    store.update_status("done", "completed")

    assert store.mark_interrupted_runs(stale_after_seconds=0.0) == ["stale"]
    assert store.read_record("stale").status == "interrupted"  # type: ignore[union-attr]
    assert store.read_record("done").status == "completed"  # type: ignore[union-attr]


# -- durable runner --


async def test_durable_run_writes_transcript_and_checkpoints(tmp_path: Path) -> None:
    store = FilesystemRunStore(tmp_path)
    adapter = FakeAdapter(
        responses=[
            FakeResponse(text="working", tool_calls=(("echo", {}),)),
            FakeResponse(text="done"),
        ]
    )
    runner = _runner(adapter, store)
    result = await runner.run([LLMMessage(role="user", content=[TextBlock(text="go")])])
    assert result.stop_reason is not None and result.stop_reason.code == "success"

    records = store.list_runs()
    assert len(records) == 1
    record = records[0]
    assert record.status == "completed"

    kinds = [e.kind for e in store.read_events(record.run_id)]
    assert "message" in kinds
    assert "tool_result" in kinds
    assert "lifecycle" in kinds
    assert kinds[-1] == "stop"

    checkpoint = store.load_latest_checkpoint(record.run_id)
    assert checkpoint is not None
    assert checkpoint.turn == result.turns
    assert checkpoint.budget.turns_used == result.turns
    assert checkpoint.lifecycle_events  # observability state persisted


async def test_kill_and_resume_continues_run(tmp_path: Path) -> None:
    """Crash mid-run, restart from the latest checkpoint, finish without repeats."""
    store = FilesystemRunStore(tmp_path)

    class _CrashAfterTwoTurns(FakeAdapter):
        async def chat(self, messages, options):  # type: ignore[no-untyped-def]
            if self._cursor >= 2:
                raise RuntimeError("simulated process crash")
            return await super().chat(messages, options)

    crashing = _CrashAfterTwoTurns(
        responses=[
            FakeResponse(text="step 1", tool_calls=(("echo", {"n": 1}),)),
            FakeResponse(text="step 2", tool_calls=(("echo", {"n": 2}),)),
            FakeResponse(text="never reached"),
        ]
    )
    runner = _runner(crashing, store)
    events = [ev async for ev in runner.stream([LLMMessage(role="user", content=[TextBlock(text="go")])])]
    assert any(ev.type == "error" for ev in events)

    record = store.list_runs()[0]
    assert record.status == "failed"

    checkpoint = store.load_latest_checkpoint(record.run_id)
    assert checkpoint is not None
    # The crashed turn counts as attempted (turn 3) but produced nothing durable:
    # the conversation holds exactly the two completed assistant turns.
    assert checkpoint.turn == 3
    tool_turns = [m for m in checkpoint.messages if m.role == "assistant"]
    assert len(tool_turns) == 2

    # "Restart the process": fresh store instance, fresh adapter continuing the script.
    fresh_store = FilesystemRunStore(tmp_path)
    restored = fresh_store.load_latest_checkpoint(record.run_id)
    assert restored is not None
    continuing = FakeAdapter(responses=[FakeResponse(text="finished")])
    resumed_runner = _runner(continuing, fresh_store, resume=restored)

    result = await resumed_runner.run([])
    assert result.stop_reason is not None and result.stop_reason.code == "success"
    assert result.output == "finished"
    # Turn count and usage continue from the checkpoint instead of resetting.
    assert result.turns == 4
    assert result.token_usage.input_tokens > restored.token_usage.input_tokens
    # No completed tool call was repeated after resume.
    assert result.tool_calls == []
    assert fresh_store.read_record(record.run_id).status == "completed"  # type: ignore[union-attr]


# -- checkpoint serializer v2 --


def _sample_agent_result() -> AgentRunResult:
    return AgentRunResult(
        success=True,
        output="ok",
        messages=[LLMMessage(role="assistant", content=[TextBlock(text="ok")])],
        token_usage=TokenUsage(input_tokens=1, output_tokens=2),
        tool_calls=[],
        terminal_phase="completed",
        stop_reason=StopReason(code="success", message="done", recoverable=False),
        retries=1,
        lifecycle_events=[LifecycleEvent(run_id="r", agent_name="a", phase="completed")],
        verification_results=[
            VerificationResult(sensor_name="ruff", kind="computational", passed=True, severity="info", message="clean")
        ],
        gate_decisions=[QualityGateDecision(outcome="pass", message="ok", results=())],
    )


def test_checkpoint_v2_roundtrips_observability_state() -> None:
    data = CheckpointData(
        id="cp-1",
        workflow_id="wf-1",
        tasks=[],
        agent_results={"a": _sample_agent_result()},
        wave_index=0,
        total_token_usage=TokenUsage(input_tokens=1, output_tokens=2),
        created_at=datetime.now(UTC),
    )
    restored = deserialize_checkpoint(serialize_checkpoint(data))
    result = restored.agent_results["a"]
    assert result.terminal_phase == "completed"
    assert result.stop_reason is not None and result.stop_reason.code == "success"
    assert result.retries == 1
    assert result.lifecycle_events[0].phase == "completed"
    assert result.verification_results[0].sensor_name == "ruff"
    assert result.gate_decisions[0].outcome == "pass"


def test_checkpoint_v1_payload_still_loads() -> None:
    """A pre-v2 checkpoint (no observability fields) deserializes with defaults."""
    v1_raw = """
    {
      "id": "cp-old", "workflow_id": "wf-old", "version": 1, "wave_index": 0,
      "total_token_usage": {"input_tokens": 1, "output_tokens": 1},
      "created_at": "2026-01-01T00:00:00+00:00",
      "metadata": null,
      "tasks": [],
      "agent_results": {
        "a": {
          "success": true, "output": "legacy",
          "token_usage": {"input_tokens": 1, "output_tokens": 1},
          "tool_calls": [], "messages": []
        }
      }
    }
    """
    restored = deserialize_checkpoint(v1_raw)
    assert restored.version == 1
    result = restored.agent_results["a"]
    assert result.output == "legacy"
    assert result.lifecycle_events == []
    assert result.stop_reason is None


async def test_budget_state_survives_resume(tmp_path: Path) -> None:
    """Budget counters continue across restart instead of resetting."""
    store = FilesystemRunStore(tmp_path)
    adapter = FakeAdapter(responses=[FakeResponse(text="a", tool_calls=(("echo", {}),)), FakeResponse(text="b")])
    runner = _runner(adapter, store)
    await runner.run([LLMMessage(role="user", content=[TextBlock(text="go")])])

    record = store.list_runs()[0]
    checkpoint = store.load_latest_checkpoint(record.run_id)
    assert checkpoint is not None

    resumed = _runner(FakeAdapter(responses=[FakeResponse(text="c")]), store, resume=checkpoint)
    assert resumed.budget_tracker.turns_used == 0  # not yet restored (restore happens on stream start)
    await resumed.run([])
    assert resumed.budget_tracker.turns_used == checkpoint.budget.turns_used + 1
    assert resumed.budget_tracker.tokens_used > checkpoint.budget.tokens_used
