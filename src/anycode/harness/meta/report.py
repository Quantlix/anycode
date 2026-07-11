"""Persistence and rendering helpers for :class:`MetaHarnessReport`."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import TypedDict

from anycode.security.redaction import redact_sensitive, redact_text
from anycode.types import MetaHarnessReport


class BlueprintRankEntry(TypedDict):
    """One row of a :func:`compare_blueprints` ranking."""

    blueprint_id: str
    train_mean: float
    heldout_mean: float
    regression_rate: float
    cost_usd: float
    accepted_changes: int
    rejected_changes: int


class BlueprintComparison(TypedDict):
    """Structured output of :func:`compare_blueprints`."""

    winner: str | None
    ranking: list[BlueprintRankEntry]


def _mean(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def save_meta_report(report: MetaHarnessReport, path: str | Path, *, redact_sensitive_data: bool = True) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    if redact_sensitive_data:
        payload = redact_sensitive(payload)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return target


def render_meta_report(report: MetaHarnessReport, *, redact_sensitive_data: bool = True) -> str:
    notes = redact_text(report.notes) if redact_sensitive_data else report.notes
    lines = [
        f"# Meta Harness Report: {report.blueprint_id}",
        "",
        f"- Train score (mean): **{_mean(report.train_scores):.4f}** over {len(report.train_scores)} runs",
        f"- Held-out score (mean): **{_mean(report.heldout_scores):.4f}** over {len(report.heldout_scores)} runs",
        f"- Accepted changes: **{report.accepted_changes}**",
        f"- Rejected changes: **{report.rejected_changes}**",
        f"- Regression rate: **{report.regression_rate:.4f}**",
        f"- Total cost: **${report.total_cost_usd:.4f}**",
        f"- Convergence iterations: {list(report.convergence_iterations)}",
    ]
    if notes:
        lines.extend(["", "## Notes", "", notes])
    return "\n".join(lines)


def compare_blueprints(reports: Sequence[MetaHarnessReport]) -> BlueprintComparison:
    """Rank a sequence of :class:`MetaHarnessReport` by held-out score then train score."""

    ranked = sorted(
        reports,
        key=lambda r: (_mean(r.heldout_scores), _mean(r.train_scores)),
        reverse=True,
    )
    summary: list[BlueprintRankEntry] = [
        {
            "blueprint_id": r.blueprint_id,
            "train_mean": _mean(r.train_scores),
            "heldout_mean": _mean(r.heldout_scores),
            "regression_rate": r.regression_rate,
            "cost_usd": r.total_cost_usd,
            "accepted_changes": r.accepted_changes,
            "rejected_changes": r.rejected_changes,
        }
        for r in ranked
    ]
    return {
        "winner": ranked[0].blueprint_id if ranked else None,
        "ranking": summary,
    }


__all__ = [
    "BlueprintComparison",
    "BlueprintRankEntry",
    "compare_blueprints",
    "render_meta_report",
    "save_meta_report",
]
