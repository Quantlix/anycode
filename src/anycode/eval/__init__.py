"""Harness benchmark and evaluation suite."""

from anycode.eval.report import compare_reports, read_report, render_markdown, write_report
from anycode.eval.scenario import load_scenario, load_scenarios
from anycode.eval.scorer import score
from anycode.eval.suite import build_agent, detect_provider, run_scenario, run_suite

__all__ = [
    "build_agent",
    "compare_reports",
    "detect_provider",
    "load_scenario",
    "load_scenarios",
    "read_report",
    "render_markdown",
    "run_scenario",
    "run_suite",
    "score",
    "write_report",
]
