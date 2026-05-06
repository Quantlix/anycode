"""Score a real agent run against an EvalScenario."""

from __future__ import annotations

from anycode.types import AgentRunResult, EvalScenario, EvalScenarioResult, TokenUsage


def score(
    scenario: EvalScenario,
    result: AgentRunResult | None,
    *,
    runtime_seconds: float,
    failure_reason: str | None = None,
) -> EvalScenarioResult:
    if result is None:
        return EvalScenarioResult(
            scenario_name=scenario.name,
            passed=False,
            output="",
            stop_reason_code=None,
            expected_stop_reason=scenario.expected_stop_reason,
            runtime_seconds=runtime_seconds,
            turns=0,
            tool_calls=0,
            failure_reason=failure_reason or "agent did not produce a result",
            missing_criteria=scenario.success_criteria,
        )

    output = result.output or ""
    matched: list[str] = []
    missing: list[str] = []
    for needle in scenario.success_criteria:
        if needle.lower() in output.lower():
            matched.append(needle)
        else:
            missing.append(needle)

    forbidden_hits: list[str] = [n for n in scenario.forbidden_substrings if n.lower() in output.lower()]

    stop_code = result.stop_reason.code if result.stop_reason else None
    stop_ok = scenario.expected_stop_reason is None or stop_code == scenario.expected_stop_reason
    criteria_ok = not missing
    forbidden_ok = not forbidden_hits
    passed = bool(stop_ok and criteria_ok and forbidden_ok and result.success)

    failure: str | None = None
    if not passed:
        parts: list[str] = []
        if not result.success:
            parts.append("agent reported failure")
        if not stop_ok:
            parts.append(f"stop_reason expected={scenario.expected_stop_reason!r} got={stop_code!r}")
        if not criteria_ok:
            parts.append(f"missing criteria: {missing}")
        if not forbidden_ok:
            parts.append(f"forbidden substrings present: {forbidden_hits}")
        failure = "; ".join(parts) or "unknown failure"

    return EvalScenarioResult(
        scenario_name=scenario.name,
        passed=passed,
        output=output,
        stop_reason_code=stop_code,
        expected_stop_reason=scenario.expected_stop_reason,
        runtime_seconds=runtime_seconds,
        turns=len([m for m in result.messages if m.role == "assistant"]),
        tool_calls=len(result.tool_calls),
        token_usage=result.token_usage or TokenUsage(),
        failure_reason=failure,
        matched_criteria=tuple(matched),
        missing_criteria=tuple(missing),
    )
