"""Tests for the `anycode runs` operator CLI over the durable run store."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

typer_testing = pytest.importorskip("typer.testing")

from anycode.cli.main import app  # noqa: E402
from anycode.runstore.store import FilesystemRunStore  # noqa: E402
from anycode.types import BudgetSnapshot, LLMMessage, TextBlock, TokenUsage, TurnCheckpoint  # noqa: E402

runner = typer_testing.CliRunner()


def _seed_run(root: Path, run_id: str = "run-1") -> FilesystemRunStore:
    store = FilesystemRunStore(root)
    store.create_run(run_id, agent_name="worker", model="fake-model")
    store.append_event(run_id, "message", {"role": "assistant"})
    store.append_event(run_id, "tool_result", {"tool_name": "bash", "output": "ok"})
    store.append_event(run_id, "stop", {"code": "success", "message": "done"})
    store.save_checkpoint(
        TurnCheckpoint(
            run_id=run_id,
            turn=3,
            messages=[LLMMessage(role="user", content=[TextBlock(text="hi")])],
            token_usage=TokenUsage(input_tokens=120, output_tokens=45),
            budget=BudgetSnapshot(tokens_used=165, cost_used=0.0123, turns_used=3, tool_calls_used=1),
            created_at=datetime.now(UTC),
        )
    )
    store.update_status(run_id, "completed")
    return store


def test_runs_list_shows_status_and_cost(tmp_path: Path) -> None:
    _seed_run(tmp_path)
    result = runner.invoke(app, ["runs", "list", "--root", str(tmp_path)])
    assert result.exit_code == 0
    assert "run-1" in result.output
    assert "completed" in result.output
    assert "0.0123" in result.output


def test_runs_show_prints_events_and_accounting(tmp_path: Path) -> None:
    _seed_run(tmp_path)
    result = runner.invoke(app, ["runs", "show", "run-1", "--root", str(tmp_path)])
    assert result.exit_code == 0
    assert "status: completed" in result.output
    assert "120 in / 45 out tokens" in result.output
    assert "tool_result" in result.output
    assert "success: done" in result.output


def test_runs_show_unknown_run_fails_cleanly(tmp_path: Path) -> None:
    result = runner.invoke(app, ["runs", "show", "ghost", "--root", str(tmp_path)])
    assert result.exit_code == 1
    assert "No run 'ghost'" in result.output


def test_runs_tail_respects_after_seq(tmp_path: Path) -> None:
    _seed_run(tmp_path)
    result = runner.invoke(app, ["runs", "tail", "run-1", "--root", str(tmp_path), "--after", "2"])
    assert result.exit_code == 0
    assert "stop" in result.output
    assert "message" not in result.output.replace("stop", "")


def test_runs_audit_digest_is_deterministic(tmp_path: Path) -> None:
    _seed_run(tmp_path)
    first = runner.invoke(app, ["runs", "audit", "run-1", "--root", str(tmp_path)])
    second = runner.invoke(app, ["runs", "audit", "run-1", "--root", str(tmp_path)])
    assert first.exit_code == 0
    assert "3 events" in first.output
    assert "bash" in first.output  # tools-used section
    assert "stop: success" in first.output or "success: done" in first.output
    assert first.output == second.output  # no LLM, no randomness


def test_runs_sweep_reports_watchdog_actions(tmp_path: Path) -> None:
    store = FilesystemRunStore(tmp_path)
    store.create_run("crashed", agent_name="a", model="m")  # running + stale heartbeat
    result = runner.invoke(app, ["runs", "sweep", "--root", str(tmp_path), "--stale-after", "0"])
    assert result.exit_code == 0
    assert "crashed" in result.output
    assert store.read_record("crashed").status == "interrupted"  # type: ignore[union-attr]
