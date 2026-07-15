"""End-to-end checks for deterministic runtime contract examples."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def _run_example(name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples" / name)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_lifecycle_contract_example_preserves_team_gate_evidence() -> None:
    completed = _run_example("35_lifecycle_contract.py")

    assert completed.stdout.splitlines() == [
        "success=False",
        "stop_reason=verification_failed",
        "gate_outcome=block",
        "sensor=regex",
    ]


def test_runtime_baseline_covers_required_metrics() -> None:
    payload = json.loads(_run_example("36_runtime_baseline.py").stdout)

    assert payload["schema_version"] == 1
    metrics = payload["metrics"]
    assert set(metrics) == {"task_admission", "execution", "checkpoint_size", "event_volume", "context_growth"}
    assert metrics["task_admission"]["tasks"] == 250
    assert metrics["execution"]["runs"] == 20
    assert metrics["checkpoint_size"]["tasks"] == 25
    assert metrics["checkpoint_size"]["bytes"] > 0
    assert metrics["event_volume"]["stream_events_per_run"] > 0
    assert metrics["event_volume"]["lifecycle_events_per_run"] > 0

    context = metrics["context_growth"]["measurements"]
    assert [measurement["messages"] for measurement in context] == [1, 8, 32]
    tokens = [measurement["estimated_tokens"] for measurement in context]
    assert tokens == sorted(tokens)
    assert len(set(tokens)) == len(tokens)
