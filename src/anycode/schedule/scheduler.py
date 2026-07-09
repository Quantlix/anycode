"""Run-store sweeps: wake due runs, mark crashed runs, warn on stalls.

Months-scale operation is durable state plus scheduled resumption, not one
process staying alive for months. The sweep is the wake mechanism, and it has
two deployment shapes with identical semantics:

* **External** (cron / systemd timer / Task Scheduler): call `sweep_once` from
  a short-lived process (`anycode runs sweep`). Idempotent and safe to run
  concurrently — a per-run lock prevents double-resume.
* **In-process** (`RunScheduler.run`): an asyncio tick loop for long-lived
  embeddings that sweeps on an interval.

Watchdog semantics: a stale heartbeat means the process died — mark the run
`interrupted`. A fresh heartbeat with no recent progress events is only a
`stall_warning` on the audit stream, never an automatic kill: a run blocked on
a human approval for three days is healthy, not hung.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from anycode.runstore.store import FilesystemRunStore

ResumeFn = Callable[[str], Awaitable[None]]
"""Resumes one run by id (typically: load latest checkpoint, run to completion)."""


class SweepReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    woken: tuple[str, ...] = ()
    interrupted: tuple[str, ...] = ()
    stalled: tuple[str, ...] = ()


async def sweep_once(
    store: FilesystemRunStore,
    *,
    resume: ResumeFn | None = None,
    stale_after_seconds: float = 600.0,
    stall_after_seconds: float = 900.0,
    tolerance_seconds: float = 30.0,
) -> SweepReport:
    """One idempotent pass over the run store: crash detection, stall warnings,
    and due-wake resumption."""
    interrupted = tuple(store.mark_interrupted_runs(stale_after_seconds))
    stalled = tuple(_warn_stalled_runs(store, stall_after_seconds))

    woken: list[str] = []
    if resume is not None:
        for record in store.due_wakes(tolerance_seconds=tolerance_seconds):
            if not store.try_acquire_sweep_lock(record.run_id):
                continue  # another sweep owns this run
            try:
                # Re-check under the lock: a concurrent sweep may have resumed it.
                current = store.read_record(record.run_id)
                if current is None or current.status != "paused":
                    continue
                store.append_event(record.run_id, "wake", {"kind": record.wake.kind if record.wake else "manual"})
                store.update_status(record.run_id, "running")
                await resume(record.run_id)
                woken.append(record.run_id)
            finally:
                store.release_sweep_lock(record.run_id)

    return SweepReport(woken=tuple(woken), interrupted=interrupted, stalled=stalled)


def _warn_stalled_runs(store: FilesystemRunStore, stall_after_seconds: float) -> list[str]:
    now = datetime.now(UTC)
    stalled: list[str] = []
    for record in store.list_runs():
        if record.status != "running":
            continue
        events = store.read_events(record.run_id)
        if not events:
            continue
        last = events[-1]
        if last.kind == "stall_warning":
            continue  # already warned; don't spam the audit stream
        idle_seconds = (now - last.ts).total_seconds()
        if idle_seconds > stall_after_seconds:
            store.append_event(
                record.run_id,
                "stall_warning",
                {"idle_seconds": int(idle_seconds), "last_event_kind": last.kind},
            )
            stalled.append(record.run_id)
    return stalled


class RunScheduler:
    """In-process tick loop over `sweep_once` for long-lived embeddings."""

    def __init__(
        self,
        store: FilesystemRunStore,
        *,
        resume: ResumeFn | None = None,
        interval_seconds: float = 1.0,
        stale_after_seconds: float = 600.0,
        stall_after_seconds: float = 900.0,
    ) -> None:
        self._store = store
        self._resume = resume
        self._interval = interval_seconds
        self._stale_after = stale_after_seconds
        self._stall_after = stall_after_seconds
        self._stopped = asyncio.Event()

    def stop(self) -> None:
        self._stopped.set()

    async def run(self) -> None:
        while not self._stopped.is_set():
            await sweep_once(
                self._store,
                resume=self._resume,
                stale_after_seconds=self._stale_after,
                stall_after_seconds=self._stall_after,
            )
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=self._interval)
            except TimeoutError:
                continue
