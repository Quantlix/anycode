"""Operator surface over the durable run store: list, show, tail, audit, sweep.

A run you cannot audit after 24 unattended hours is a run you cannot trust for
24 unattended hours. Everything here is a deterministic view over the same
append-only transcript the runner writes — there is no second log to disagree
with. Rendering follows "success is silent, failures are verbose": routine
events compress to counts, stops and warnings print in full.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import UTC, datetime, timedelta

import typer
from rich.console import Console
from rich.table import Table

from anycode.runstore.store import FilesystemRunStore
from anycode.types import RunRetentionPolicy

app = typer.Typer(no_args_is_help=True)
console = Console()

_ROOT_OPTION = typer.Option(".anycode/runs", "--root", help="Run store root directory.")


@app.command("list", help="List durable runs with status, age, and cost so far.")
def list_runs(root: str = _ROOT_OPTION) -> None:
    store = FilesystemRunStore(root)
    records = store.list_runs()
    if not records:
        console.print("No durable runs found.")
        raise typer.Exit(0)

    table = Table(title="Durable runs")
    for column in ("run id", "agent", "model", "status", "updated", "turns", "cost ($)"):
        table.add_column(column)
    for record in records:
        checkpoint = store.load_latest_checkpoint(record.run_id)
        turns = str(checkpoint.turn) if checkpoint else "-"
        cost = f"{checkpoint.budget.cost_used:.4f}" if checkpoint else "-"
        table.add_row(
            record.run_id,
            record.agent_name,
            record.model,
            record.status,
            record.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
            turns,
            cost,
        )
    console.print(table)


@app.command("show", help="Show one run: status, wake condition, accounting, and recent events.")
def show(run_id: str, root: str = _ROOT_OPTION, events: int = typer.Option(10, help="Recent events to print.")) -> None:
    store = FilesystemRunStore(root)
    record = store.read_record(run_id)
    if record is None:
        console.print(f"[red]No run '{run_id}' under {root}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]{record.run_id}[/bold]  agent={record.agent_name}  model={record.model}")
    console.print(f"status: {record.status}   heartbeat: {record.last_heartbeat.isoformat()}")
    if record.wake is not None:
        console.print(f"wake: {record.wake.kind} at {record.wake.wake_at}  ({record.wake.note})")

    checkpoint = store.load_latest_checkpoint(run_id)
    if checkpoint is not None:
        usage = checkpoint.token_usage
        console.print(
            f"turn {checkpoint.turn}: {usage.input_tokens} in / {usage.output_tokens} out tokens, "
            f"${checkpoint.budget.cost_used:.4f} spent, {checkpoint.budget.tool_calls_used} tool calls"
        )

    recent = store.read_events(run_id)[-events:]
    for event in recent:
        console.print(f"  #{event.seq:<5} {event.ts.strftime('%H:%M:%S')} {event.kind:<14} {_event_line(event.kind, event.payload)}")


@app.command("tail", help="Print transcript events after a sequence number (for follow-up polling).")
def tail(run_id: str, root: str = _ROOT_OPTION, after: int = typer.Option(0, help="Only events after this seq.")) -> None:
    store = FilesystemRunStore(root)
    for event in store.read_events(run_id, after_seq=after):
        console.print(f"#{event.seq:<5} {event.ts.isoformat()} {event.kind:<14} {_event_line(event.kind, event.payload)}")


@app.command("audit", help="Deterministic digest of a run's activity within a window.")
def audit(
    run_id: str,
    root: str = _ROOT_OPTION,
    since_hours: float = typer.Option(24.0, "--since-hours", help="Window to audit."),
) -> None:
    store = FilesystemRunStore(root)
    record = store.read_record(run_id)
    if record is None:
        console.print(f"[red]No run '{run_id}' under {root}[/red]")
        raise typer.Exit(1)

    cutoff = datetime.now(UTC) - timedelta(hours=since_hours)
    window = [e for e in store.read_events(run_id) if e.ts >= cutoff]

    console.print(f"[bold]Audit: {run_id}[/bold] — {len(window)} events in the last {since_hours:g}h")
    counts = Counter(e.kind for e in window)
    for kind in sorted(counts):
        console.print(f"  {kind:<16} {counts[kind]}")

    tools = Counter(str(e.payload.get("tool_name", "?")) for e in window if e.kind == "tool_result")
    if tools:
        console.print("tools used:")
        for name, count in tools.most_common():
            console.print(f"  {name:<16} {count}")

    for event in window:
        if event.kind in ("stop", "stall_warning", "pause", "wake"):
            console.print(f"  ! #{event.seq} {event.kind}: {_event_line(event.kind, event.payload)}")


@app.command("sweep", help="One watchdog pass: mark crashed runs, warn stalls, report due wakes.")
def sweep(
    root: str = _ROOT_OPTION,
    stale_after: float = typer.Option(600.0, help="Heartbeat staleness (s) before a running run counts as crashed."),
    stall_after: float = typer.Option(900.0, help="Idle window (s) before a live run gets a stall warning."),
    retention_days: float | None = typer.Option(None, help="Delete terminal runs older than this many days."),
    max_runs: int | None = typer.Option(None, help="Keep at most this many terminal runs."),
) -> None:
    from anycode.schedule.scheduler import sweep_once

    store = FilesystemRunStore(root)
    retention_policy = (
        RunRetentionPolicy(max_age_days=retention_days, max_runs=max_runs) if retention_days is not None or max_runs is not None else None
    )
    report = asyncio.run(
        sweep_once(
            store,
            stale_after_seconds=stale_after,
            stall_after_seconds=stall_after,
            retention_policy=retention_policy,
        )
    )
    due = store.due_wakes()
    console.print(f"interrupted: {list(report.interrupted) or '-'}")
    console.print(f"stalled:     {list(report.stalled) or '-'}")
    console.print(f"pruned:      {list(report.pruned) or '-'}")
    if due:
        console.print("due wakes (resume programmatically via AgentRunner(resume_from=...) or RunScheduler):")
        for record in due:
            console.print(f"  {record.run_id}  wake={record.wake.kind if record.wake else '?'} at {record.wake.wake_at if record.wake else '?'}")
    else:
        console.print("due wakes:   -")


def _event_line(kind: str, payload: dict[str, object]) -> str:
    if kind == "tool_result":
        return f"{payload.get('tool_name', '?')} ({len(str(payload.get('output', '')))} chars)"
    if kind == "stop":
        return f"{payload.get('code', '?')}: {payload.get('message', '')}"
    if kind == "lifecycle":
        return str(payload.get("phase", ""))
    if kind == "checkpoint":
        return f"turn {payload.get('turn', '?')}"
    if kind in ("pause", "wake"):
        return str(payload.get("kind", ""))
    if kind == "compaction":
        return f"pressure={payload.get('pressure', '')} reset={payload.get('reset', False)}"
    return ""
