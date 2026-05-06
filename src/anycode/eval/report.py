"""EvalReport persistence and comparison helpers."""

from __future__ import annotations

import json
from pathlib import Path

from anycode.types import EvalReport


def write_report(report: EvalReport, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.model_dump(), indent=2, default=str), encoding="utf-8")
    return target


def read_report(path: str | Path) -> EvalReport:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalReport.model_validate(payload)


def render_markdown(report: EvalReport) -> str:
    lines = [
        f"# Eval Report: {report.suite_name} ({report.harness_variant})",
        "",
        f"- Scenarios: **{report.total_scenarios}**",
        f"- Passed: **{report.passed}**",
        f"- Failed: **{report.failed}**",
        f"- Total runtime: **{report.total_runtime_seconds:.3f}s**",
        f"- Tokens: input={report.total_input_tokens} output={report.total_output_tokens}",
        f"- Cost: **${report.total_cost_usd:.4f}**",
        f"- Retries: {report.total_retries}",
        f"- Verification failures: {report.total_verification_failures}",
        "",
        "| Scenario | Passed | Stop Reason | Runtime (s) | Turns | Tools | Cost ($) | Retries | VFails | Failure |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report.scenario_results:
        lines.append(
            f"| {r.scenario_name} | {'\u2714' if r.passed else '\u2718'} | {r.stop_reason_code or '-'} | "
            f"{r.runtime_seconds:.3f} | {r.turns} | {r.tool_calls} | "
            f"{r.cost_usd:.4f} | {r.retries} | {r.verification_failures} | {r.failure_reason or ''} |"
        )
    return "\n".join(lines)


def compare_reports(baseline: EvalReport, candidate: EvalReport) -> dict:
    """Return a structured diff of two reports."""
    baseline_by_name = {r.scenario_name: r for r in baseline.scenario_results}
    candidate_by_name = {r.scenario_name: r for r in candidate.scenario_results}

    regressions: list[str] = []
    improvements: list[str] = []
    for name, base in baseline_by_name.items():
        cand = candidate_by_name.get(name)
        if cand is None:
            regressions.append(f"{name}: missing from candidate")
            continue
        if base.passed and not cand.passed:
            regressions.append(f"{name}: passed in baseline, failed in candidate")
        elif not base.passed and cand.passed:
            improvements.append(f"{name}: failed in baseline, passed in candidate")

    new_scenarios = [name for name in candidate_by_name if name not in baseline_by_name]

    return {
        "baseline": {"variant": baseline.harness_variant, "passed": baseline.passed, "failed": baseline.failed},
        "candidate": {"variant": candidate.harness_variant, "passed": candidate.passed, "failed": candidate.failed},
        "regressions": regressions,
        "improvements": improvements,
        "new_scenarios": new_scenarios,
        "runtime_delta_seconds": candidate.total_runtime_seconds - baseline.total_runtime_seconds,
    }
