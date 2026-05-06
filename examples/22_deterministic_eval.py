"""Phase 6 — deterministic evaluation suite (no LLM key required).

Runs the deterministic harness-runtime-reliability fixture using the
``FakeAdapter`` so the scenarios never call a real provider. Produces JSON +
Markdown reports next to the fixture so users can inspect cost/retry/
verification-failure aggregates introduced in Phase 6.

Run::

    uv run python examples/22_deterministic_eval.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from anycode import load_scenarios, render_markdown, run_suite, write_report

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "eval" / "runtime_reliability_deterministic.yaml"


async def main() -> None:
    scenarios = list(load_scenarios(FIXTURE))
    print(f"Loaded {len(scenarios)} deterministic scenarios from {FIXTURE.name}")

    report = await run_suite(scenarios, suite_name="phase6", harness_variant="deterministic")

    print("\n=== Aggregate metrics ===")
    print(f"  passed:                     {report.passed}/{report.total_scenarios}")
    print(f"  total_runtime_seconds:      {report.total_runtime_seconds:.3f}")
    print(f"  total_cost_usd:             ${report.total_cost_usd:.6f}")
    print(f"  total_retries:              {report.total_retries}")
    print(f"  total_verification_failures:{report.total_verification_failures}")

    out_dir = Path(tempfile.mkdtemp(prefix="anycode-phase6-"))
    json_path = out_dir / "phase6_eval.json"
    md_path = out_dir / "phase6_eval.md"
    write_report(report, json_path)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"\nReports written to {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
