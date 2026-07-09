"""Scheduled wakeups: pause a run for later, wake it with a sweep.

Months-scale operation is durable state plus scheduled resumption — not one
process staying alive for months. A paused run persists its wake condition;
any later process (cron, systemd timer, `anycode runs sweep`) sweeps the run
store, wakes due runs, marks crashed ones, and warns about stalls.

Scheduled task modes keep token spend proportional to reasoning: notification
and script modes cost zero LLM calls; hybrid invokes the agent only when the
script's output demands interpretation.

Run::

    uv run python examples/30_scheduled_wakeups.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from datetime import UTC, datetime, timedelta

from anycode import FilesystemRunStore, ScheduledTask, WakeCondition, run_scheduled_task, sweep_once


async def main() -> None:
    root = tempfile.mkdtemp(prefix="anycode-schedule-")
    store = FilesystemRunStore(root)

    # --- pause a run with a timed wake, "exit the process" ----------------
    store.create_run("nightly-report", agent_name="reporter", model="fake-model")
    store.pause_run(
        "nightly-report",
        WakeCondition(kind="at_time", wake_at=datetime.now(UTC) - timedelta(seconds=1), note="resume after window"),
    )
    print(f"paused: {store.read_record('nightly-report').status}")  # type: ignore[union-attr]

    # --- later, from any process: one sweep wakes what is due -------------
    async def resume(run_id: str) -> None:
        # Real deployments load the latest checkpoint and continue the run
        # (AgentRunner(resume_from=...)); here we just complete it.
        store.update_status(run_id, "completed")
        print(f"resumed {run_id} from its persisted wake condition")

    report = await sweep_once(FilesystemRunStore(root), resume=resume)
    print(f"sweep report: woken={list(report.woken)}, interrupted={list(report.interrupted)}")

    # --- scheduled task modes: spend tokens only on judgment --------------
    async def agent(prompt: str) -> str:
        return f"[agent reasoned over {len(prompt)} chars of script output]"

    py = sys.executable.replace("\\", "/")
    healthy = await run_scheduled_task(
        ScheduledTask(name="disk-check", mode="hybrid", command=f'"{py}" -c "print(\'disk ok\')"', prompt="diagnose"),
        agent=agent,
    )
    failing = await run_scheduled_task(
        ScheduledTask(
            name="disk-check",
            mode="hybrid",
            command=f'"{py}" -c "import sys; print(\'disk 97% full\'); sys.exit(1)"',
            prompt="diagnose the disk alert",
        ),
        agent=agent,
    )
    print(f"healthy check: agent_invoked={healthy.agent_invoked} (deterministic, 0 LLM calls)")
    print(f"failing check: agent_invoked={failing.agent_invoked} -> {failing.output}")

    assert report.woken == ("nightly-report",)
    assert not healthy.agent_invoked and failing.agent_invoked


if __name__ == "__main__":
    asyncio.run(main())
