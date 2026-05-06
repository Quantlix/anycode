"""Harness evaluation suite demo (real LLM only).

Loads ``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY`` from .env, runs the YAML
scenario fixture against the live provider, persists JSON+Markdown reports to
``artifacts/eval/``, and demonstrates report comparison between two runs.

Run::

    uv run python examples/21_eval_suite.py
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from anycode import (
    compare_reports,
    detect_provider,
    load_scenarios,
    read_report,
    render_markdown,
    run_suite,
    write_report,
)

ROOT = Path(__file__).resolve().parent.parent
SCENARIO_FILE = ROOT / "tests" / "fixtures" / "eval" / "runtime_reliability.yaml"
ARTIFACT_DIR = ROOT / "artifacts" / "eval"


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


async def _run_baseline() -> Path:
    print("\n== Section 1: Run suite against real LLM ==")
    provider, model = detect_provider()
    print(f"Provider: {provider}  Model: {model}")
    scenarios = load_scenarios(SCENARIO_FILE)
    print(f"Loaded {len(scenarios)} scenarios from {SCENARIO_FILE.name}")

    report = await run_suite(scenarios, suite_name="runtime_reliability", harness_variant="baseline")
    print(
        f"Pass: {report.passed}/{report.total_scenarios}  "
        f"Runtime: {report.total_runtime_seconds:.2f}s  "
        f"Tokens: in={report.total_input_tokens} out={report.total_output_tokens}"
    )

    target = ARTIFACT_DIR / f"{_stamp()}_baseline.json"
    write_report(report, target)
    md_target = target.with_suffix(".md")
    md_target.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote: {target.name}  +  {md_target.name}")
    for r in report.scenario_results:
        marker = "PASS" if r.passed else "FAIL"
        print(f"  [{marker}] {r.scenario_name}  stop={r.stop_reason_code}  reason={r.failure_reason or '-'}")
    return target


async def _run_candidate(baseline_path: Path) -> None:
    print("\n== Section 2: Re-run suite as candidate variant ==")
    scenarios = load_scenarios(SCENARIO_FILE)
    report = await run_suite(scenarios, suite_name="runtime_reliability", harness_variant="candidate")
    target = ARTIFACT_DIR / f"{_stamp()}_candidate.json"
    write_report(report, target)
    print(f"Wrote: {target.name}")

    print("\n== Section 3: Compare baseline vs candidate ==")
    base = read_report(baseline_path)
    cand = read_report(target)
    diff = compare_reports(base, cand)
    print(f"Baseline: {diff['baseline']}")
    print(f"Candidate: {diff['candidate']}")
    print(f"Regressions: {diff['regressions'] or 'none'}")
    print(f"Improvements: {diff['improvements'] or 'none'}")
    print(f"New scenarios: {diff['new_scenarios'] or 'none'}")
    print(f"Runtime delta: {diff['runtime_delta_seconds']:+.2f}s")


async def main() -> None:
    load_dotenv(ROOT / ".env", override=False)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path = await _run_baseline()
    await _run_candidate(baseline_path)
    print("\nDone. Artifacts in:", ARTIFACT_DIR)


if __name__ == "__main__":
    asyncio.run(main())
