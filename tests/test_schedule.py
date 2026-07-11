"""Tests for pause/wake scheduling, watchdog sweeps, and scheduled task modes."""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from anycode.runstore.store import FilesystemRunStore
from anycode.schedule.scheduler import sweep_once
from anycode.schedule.tasks import ScheduledTask, run_scheduled_task
from anycode.types import RunRetentionPolicy, WakeCondition


def _paused_run(store: FilesystemRunStore, run_id: str, *, wake_delta_seconds: float) -> None:
    store.create_run(run_id, agent_name="a", model="m")
    store.pause_run(
        run_id,
        WakeCondition(kind="at_time", wake_at=datetime.now(UTC) + timedelta(seconds=wake_delta_seconds)),
    )


async def test_pause_persists_wake_condition(tmp_path: Path) -> None:
    store = FilesystemRunStore(tmp_path)
    _paused_run(store, "run-1", wake_delta_seconds=3600)

    record = store.read_record("run-1")
    assert record is not None
    assert record.status == "paused"
    assert record.wake is not None and record.wake.kind == "at_time"
    assert [e.kind for e in store.read_events("run-1")] == ["pause"]

    # Fresh store instance (process restart) still sees the wake condition.
    assert FilesystemRunStore(tmp_path).read_record("run-1").wake is not None  # type: ignore[union-attr]


async def test_sweep_wakes_due_runs_and_skips_future_ones(tmp_path: Path) -> None:
    store = FilesystemRunStore(tmp_path)
    _paused_run(store, "due", wake_delta_seconds=-5)
    _paused_run(store, "later", wake_delta_seconds=3600)

    resumed: list[str] = []

    async def resume(run_id: str) -> None:
        resumed.append(run_id)
        store.update_status(run_id, "completed")

    report = await sweep_once(store, resume=resume)
    assert report.woken == ("due",)
    assert resumed == ["due"]
    assert store.read_record("later").status == "paused"  # type: ignore[union-attr]

    kinds = [e.kind for e in store.read_events("due")]
    assert kinds == ["pause", "wake"]


async def test_manual_and_approval_wakes_never_autofire(tmp_path: Path) -> None:
    store = FilesystemRunStore(tmp_path)
    store.create_run("waiting", agent_name="a", model="m")
    store.pause_run("waiting", WakeCondition(kind="on_approval", approval_id="ap-1"))

    async def resume(run_id: str) -> None:  # pragma: no cover - must not be called
        raise AssertionError("approval waits must not auto-resume")

    report = await sweep_once(store, resume=resume)
    assert report.woken == ()
    assert store.read_record("waiting").status == "paused"  # type: ignore[union-attr]


async def test_concurrent_sweeps_never_double_resume(tmp_path: Path) -> None:
    store_a = FilesystemRunStore(tmp_path)
    store_b = FilesystemRunStore(tmp_path)
    _paused_run(store_a, "due", wake_delta_seconds=-5)

    resumes: list[str] = []

    def make_resume(store: FilesystemRunStore):  # type: ignore[no-untyped-def]
        async def resume(run_id: str) -> None:
            resumes.append(run_id)
            await asyncio.sleep(0.05)  # hold the lock while "resuming"
            store.update_status(run_id, "completed")

        return resume

    reports = await asyncio.gather(
        sweep_once(store_a, resume=make_resume(store_a)),
        sweep_once(store_b, resume=make_resume(store_b)),
    )
    assert resumes == ["due"], "exactly one sweep may resume the run"
    assert sum(len(r.woken) for r in reports) == 1


async def test_sweep_marks_crashed_and_warns_stalled(tmp_path: Path) -> None:
    store = FilesystemRunStore(tmp_path)
    record = store.create_run("crashed", agent_name="a", model="m")
    store._write_record(record.model_copy(update={"last_heartbeat": datetime.now(UTC) - timedelta(minutes=1)}))

    report = await sweep_once(store, stale_after_seconds=0.0, stall_after_seconds=999999)
    assert report.interrupted == ("crashed",)
    assert store.read_record("crashed").status == "interrupted"  # type: ignore[union-attr]


async def test_stall_warning_for_fresh_heartbeat_without_progress(tmp_path: Path) -> None:
    store = FilesystemRunStore(tmp_path)
    store.create_run("slow", agent_name="a", model="m")
    store.append_event("slow", "message", {})
    store.touch_heartbeat("slow")  # alive...

    report = await sweep_once(store, stale_after_seconds=3600, stall_after_seconds=0.0)
    assert report.stalled == ("slow",)
    events = store.read_events("slow")
    assert events[-1].kind == "stall_warning"
    assert store.read_record("slow").status == "running"  # type: ignore[union-attr]  # warned, never killed

    # A second sweep does not spam duplicate warnings.
    report2 = await sweep_once(store, stale_after_seconds=3600, stall_after_seconds=0.0)
    assert report2.stalled == ()


async def test_sweep_applies_explicit_retention_policy(tmp_path: Path) -> None:
    store = FilesystemRunStore(tmp_path)
    store.create_run("completed", agent_name="a", model="m")
    store.update_status("completed", "completed")
    store.create_run("active", agent_name="a", model="m")

    report = await sweep_once(
        store,
        stale_after_seconds=3600,
        stall_after_seconds=3600,
        retention_policy=RunRetentionPolicy(max_runs=0),
    )

    assert report.pruned == ("completed",)
    assert store.read_record("completed") is None
    assert store.read_record("active") is not None


# -- scheduled task modes --


async def test_notification_and_script_modes_use_zero_llm_calls() -> None:
    agent_calls: list[str] = []

    async def agent(prompt: str) -> str:  # pragma: no cover - must not be called
        agent_calls.append(prompt)
        return "agent output"

    note = await run_scheduled_task(ScheduledTask(name="ping", mode="notification", message="backup completed"), agent=agent)
    assert note.output == "backup completed"

    py = sys.executable.replace("\\", "/")
    script = await run_scheduled_task(ScheduledTask(name="check", mode="script", command=f'"{py}" -c "print(40 + 2)"'), agent=agent)
    assert script.exit_code == 0
    assert "42" in script.output

    assert agent_calls == []
    assert not note.agent_invoked and not script.agent_invoked


async def test_hybrid_mode_invokes_agent_only_on_trigger() -> None:
    prompts: list[str] = []

    async def agent(prompt: str) -> str:
        prompts.append(prompt)
        return "interpreted"

    py = sys.executable.replace("\\", "/")

    ok = await run_scheduled_task(
        ScheduledTask(name="healthy", mode="hybrid", command=f'"{py}" -c "print(1)"', prompt="diagnose"),
        agent=agent,
    )
    assert not ok.agent_invoked and prompts == []

    bad = await run_scheduled_task(
        ScheduledTask(name="failing", mode="hybrid", command=f'"{py}" -c "import sys; print(hex(255)); sys.exit(3)"', prompt="diagnose"),
        agent=agent,
    )
    assert bad.agent_invoked and bad.output == "interpreted"
    assert "0xff" in prompts[0]  # script output fed into the agent prompt


async def test_agent_mode_requires_agent() -> None:
    with pytest.raises(ValueError, match="no agent callable"):
        await run_scheduled_task(ScheduledTask(name="x", mode="agent", prompt="do things"))


async def test_provider_unavailable_pauses_durable_run(tmp_path: Path) -> None:
    """Circuit-open on a durable run parks it with a timed wake instead of failing."""
    from anycode.core.runner import AgentRunner
    from anycode.providers.resilience import ResilientAdapter
    from anycode.tools.executor import ToolExecutor
    from anycode.tools.registry import ToolRegistry
    from anycode.types import DurabilityConfig, LLMMessage, ProviderResilienceConfig, RetryPolicy, RunnerOptions, TextBlock

    class _DownAdapter:
        @property
        def name(self) -> str:
            return "down"

        async def chat(self, messages, options, **kwargs):  # type: ignore[no-untyped-def]
            err = ConnectionError("provider down")
            raise err

        def stream(self, messages, options):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    adapter = ResilientAdapter(
        _DownAdapter(),
        ProviderResilienceConfig(retry=RetryPolicy(max_attempts=1, base_delay_seconds=0.001)),
    )
    store = FilesystemRunStore(tmp_path)
    registry = ToolRegistry()
    runner = AgentRunner(
        adapter,
        registry,
        ToolExecutor(registry),
        RunnerOptions(model="fake-model", max_turns=2, agent_name="t"),
        durability=DurabilityConfig(enabled=True, run_root=str(tmp_path)),
        run_store=store,
    )
    result = await runner.run([LLMMessage(role="user", content=[TextBlock(text="go")])])
    assert result.stop_reason is not None and result.stop_reason.code == "provider_unavailable"

    record = store.list_runs()[0]
    assert record.status == "paused"
    assert record.wake is not None and record.wake.kind == "on_provider_recovery"
    assert record.wake.wake_at is not None  # timed wake -> a sweep will retry it
