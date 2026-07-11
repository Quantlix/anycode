"""Tests for the harness evaluation suite.

Live scenario execution requires a real provider key (ANTHROPIC_API_KEY or
OPENAI_API_KEY). Tests that need a live provider are skipped automatically
when no key is present so the suite stays runnable in CI without secrets.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from anycode import (
    EvalReport,
    EvalScenario,
    EvalScenarioResult,
    compare_reports,
    detect_provider,
    load_scenarios,
    read_report,
    render_markdown,
    run_scenario,
    run_suite,
    write_report,
)
from anycode.eval.scorer import score
from anycode.types import AgentRunResult, LLMMessage, StopReason, StopReasonCode, TextBlock, TokenUsage

FIXTURE = Path(__file__).parent / "fixtures" / "eval" / "runtime_reliability.yaml"

HAS_LIVE_KEY = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))
requires_live_provider = pytest.mark.skipif(not HAS_LIVE_KEY, reason="No live LLM provider key in environment")


def _make_run_result(text: str, *, success: bool = True, stop_code: StopReasonCode | None = "success") -> AgentRunResult:
    return AgentRunResult(
        success=success,
        output=text,
        messages=[LLMMessage(role="assistant", content=[TextBlock(text=text)])],
        token_usage=TokenUsage(input_tokens=5, output_tokens=10),
        tool_calls=[],
        stop_reason=StopReason(code=stop_code, message="ok") if stop_code else None,
    )


# -- Pure scoring tests (no LLM required) --


def test_score_passes_when_criteria_met() -> None:
    scenario = EvalScenario(
        name="ok",
        prompt="x",
        success_criteria=("hello",),
        expected_stop_reason="success",
    )
    result = score(scenario, _make_run_result("Hello world"), runtime_seconds=0.1)
    assert result.passed is True
    assert "hello" in result.matched_criteria


def test_score_fails_on_missing_criteria() -> None:
    scenario = EvalScenario(name="x", prompt="x", success_criteria=("nope",))
    result = score(scenario, _make_run_result("hello"), runtime_seconds=0.0)
    assert result.passed is False
    assert "nope" in result.missing_criteria
    assert "missing criteria" in (result.failure_reason or "")


def test_score_fails_on_forbidden_substring() -> None:
    scenario = EvalScenario(name="x", prompt="x", forbidden_substrings=("secret",))
    result = score(scenario, _make_run_result("here is the secret"), runtime_seconds=0.0)
    assert result.passed is False
    assert "forbidden" in (result.failure_reason or "")


def test_score_fails_on_wrong_stop_reason() -> None:
    scenario = EvalScenario(name="x", prompt="x", expected_stop_reason="max_turns")
    result = score(scenario, _make_run_result("ok", stop_code="success"), runtime_seconds=0.0)
    assert result.passed is False
    assert "stop_reason" in (result.failure_reason or "")


def test_score_handles_missing_result() -> None:
    scenario = EvalScenario(name="x", prompt="x", success_criteria=("a",))
    result = score(scenario, None, runtime_seconds=0.0, failure_reason="boom")
    assert result.passed is False
    assert result.failure_reason == "boom"
    assert result.missing_criteria == ("a",)


# -- Loading & immutability --


def test_load_scenarios_from_yaml() -> None:
    scenarios = load_scenarios(FIXTURE)
    assert len(scenarios) == 5
    names = {s.name for s in scenarios}
    assert {"arithmetic_basic", "capital_lookup", "instruction_following_format"} <= names


def test_eval_scenario_is_immutable() -> None:
    s = EvalScenario(name="x", prompt="x")
    with pytest.raises(Exception):
        s.name = "y"  # type: ignore[misc]


# -- Report persistence and comparison --


def _scenario_result(name: str, *, passed: bool) -> EvalScenarioResult:
    return EvalScenarioResult(
        scenario_name=name,
        passed=passed,
        output="ok" if passed else "no",
        runtime_seconds=0.01,
        turns=1,
        tool_calls=0,
    )


def _report(variant: str, results: tuple[EvalScenarioResult, ...]) -> EvalReport:
    passed = sum(1 for r in results if r.passed)
    return EvalReport(
        suite_name="suite",
        harness_variant=variant,
        total_scenarios=len(results),
        passed=passed,
        failed=len(results) - passed,
        total_runtime_seconds=sum(r.runtime_seconds for r in results),
        total_input_tokens=0,
        total_output_tokens=0,
        scenario_results=results,
    )


def test_report_roundtrip(tmp_path: Path) -> None:
    report = _report("v1", (_scenario_result("a", passed=True),))
    target = write_report(report, tmp_path / "out.json")
    assert target.exists()
    restored = read_report(target)
    assert restored.scenario_results[0].scenario_name == "a"


def test_report_exports_redact_secrets_by_default(tmp_path: Path) -> None:
    result = _scenario_result("secret", passed=False).model_copy(
        update={"output": "sk-1234567890abcdef1234567890", "failure_reason": "api_key=plain-value"}
    )
    report = _report("v1", (result,))

    target = write_report(report, tmp_path / "redacted.json")
    raw = target.read_text(encoding="utf-8")
    markdown = render_markdown(report)

    assert "sk-" not in raw
    assert "plain-value" not in raw
    assert "plain-value" not in markdown
    assert "plain-value" in render_markdown(report, redact_sensitive_data=False)


def test_render_markdown_contains_header_and_rows() -> None:
    report = _report("v1", (_scenario_result("a", passed=True), _scenario_result("b", passed=False)))
    md = render_markdown(report)
    assert "Eval Report" in md
    assert "✔" in md and "✘" in md
    assert "| a |" in md and "| b |" in md


def test_compare_detects_regressions_and_improvements() -> None:
    base = _report("base", (_scenario_result("a", passed=True), _scenario_result("b", passed=False)))
    cand = _report("cand", (_scenario_result("a", passed=False), _scenario_result("b", passed=True)))
    diff = compare_reports(base, cand)
    assert "a: passed in baseline, failed in candidate" in diff["regressions"]
    assert "b: failed in baseline, passed in candidate" in diff["improvements"]


def test_compare_flags_missing_scenarios() -> None:
    base = _report("base", (_scenario_result("a", passed=True), _scenario_result("b", passed=True)))
    cand = _report("cand", (_scenario_result("a", passed=True),))
    diff = compare_reports(base, cand)
    assert "b: missing from candidate" in diff["regressions"]


def test_compare_lists_new_scenarios() -> None:
    base = _report("base", (_scenario_result("a", passed=True),))
    cand = _report("cand", (_scenario_result("a", passed=True), _scenario_result("c", passed=True)))
    diff = compare_reports(base, cand)
    assert "c" in diff["new_scenarios"]


# -- Provider detection --


def test_detect_provider_raises_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError):
        detect_provider()


def test_detect_provider_picks_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider, model = detect_provider()
    assert provider == "openai"
    assert model.startswith("gpt-")


# -- Live LLM tests (skipped without keys) --


@requires_live_provider
async def test_run_scenario_against_real_provider() -> None:
    scenario = EvalScenario(
        name="capital",
        prompt="What is the capital city of France? Answer with just the city name.",
        system_prompt="You are a concise knowledge assistant. Answer in one word.",
        success_criteria=("Paris",),
        max_turns=1,
        temperature=0.0,
    )
    result = await run_scenario(scenario)
    assert result.passed, f"failed: {result.failure_reason}; output={result.output!r}"
    assert "Paris".lower() in result.output.lower()


@requires_live_provider
async def test_run_suite_against_real_provider() -> None:
    scenarios = load_scenarios(FIXTURE)
    report = await run_suite(scenarios, suite_name="runtime_reliability", harness_variant="live")
    assert report.total_scenarios == len(scenarios)
    assert report.passed >= report.total_scenarios - 1, (
        f"too many failures: {[(r.scenario_name, r.failure_reason) for r in report.scenario_results if not r.passed]}"
    )
