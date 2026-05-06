"""Tests for verification sensors and quality gates."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from anycode import (
    QualityGate,
    Sensor,
    SensorContext,
    VerificationResult,
    VerificationSensorConfig,
    decide_gate,
    schema_sensor,
)


def _ctx(phase: str = "after_task", output: str = "") -> SensorContext:
    return SensorContext(phase=phase, agent_name="agent", run_id="run-1", output=output)  # type: ignore[arg-type]


def _make_sensor(name: str, *, passed: bool, severity: str = "info") -> Sensor:
    config = VerificationSensorConfig(name=name, kind="computational", phases=("after_task",))

    def _fn(_ctx: SensorContext) -> VerificationResult:
        return VerificationResult(
            sensor_name=name,
            kind="computational",
            passed=passed,
            severity=severity,  # type: ignore[arg-type]
            message="ok" if passed else "bad",
        )

    return Sensor(config=config, fn=_fn)


def test_decide_gate_pass() -> None:
    results = [
        VerificationResult(sensor_name="a", kind="computational", passed=True, severity="info", message="ok"),
        VerificationResult(sensor_name="b", kind="computational", passed=True, severity="info", message="ok"),
    ]
    decision = decide_gate(results)
    assert decision.outcome == "pass"


def test_decide_gate_warn() -> None:
    results = [
        VerificationResult(sensor_name="a", kind="computational", passed=True, severity="info", message="ok"),
        VerificationResult(sensor_name="b", kind="computational", passed=False, severity="warning", message="meh"),
    ]
    decision = decide_gate(results)
    assert decision.outcome == "warn"


def test_decide_gate_retry_on_error() -> None:
    results = [
        VerificationResult(sensor_name="a", kind="computational", passed=False, severity="error", message="bad"),
    ]
    decision = decide_gate(results)
    assert decision.outcome == "retry"


def test_decide_gate_block_on_critical() -> None:
    results = [
        VerificationResult(sensor_name="a", kind="computational", passed=False, severity="critical", message="boom"),
    ]
    decision = decide_gate(results)
    assert decision.outcome == "block"


def test_decide_gate_escalates_after_repeated_blocks() -> None:
    results = [
        VerificationResult(sensor_name="a", kind="computational", passed=False, severity="critical", message="boom"),
    ]
    decision = decide_gate(results, escalate_on_repeated_block=True, repeated_block_count=2)
    assert decision.outcome == "escalate"


def test_decide_gate_no_results_passes() -> None:
    decision = decide_gate([])
    assert decision.outcome == "pass"
    assert decision.results == ()


async def test_quality_gate_runs_only_phase_sensors() -> None:
    after_task = _make_sensor("a", passed=True)
    other_phase = Sensor(
        config=VerificationSensorConfig(name="b", kind="computational", phases=("after_team",)),
        fn=lambda _c: VerificationResult(sensor_name="b", kind="computational", passed=False, severity="critical", message="x"),
    )
    gate = QualityGate([after_task, other_phase])
    decision = await gate.evaluate(_ctx(phase="after_task"))
    assert decision.outcome == "pass"
    assert {r.sensor_name for r in decision.results} == {"a"}


async def test_quality_gate_block_then_escalate_on_repeats() -> None:
    sensor = _make_sensor("crit", passed=False, severity="critical")
    gate = QualityGate([sensor])
    first = await gate.evaluate(_ctx())
    second = await gate.evaluate(_ctx())
    third = await gate.evaluate(_ctx())
    assert first.outcome == "block"
    assert second.outcome == "block"
    assert third.outcome == "escalate"


async def test_quality_gate_pass_resets_block_history() -> None:
    bad = _make_sensor("crit", passed=False, severity="critical")
    good = _make_sensor("ok", passed=True)
    gate = QualityGate([bad])
    first = await gate.evaluate(_ctx())
    assert first.outcome == "block"
    # Swap sensors and a passing run resets history
    gate2 = QualityGate([good])
    decision = await gate2.evaluate(_ctx())
    assert decision.outcome == "pass"


async def test_sensor_exception_becomes_failure_result() -> None:
    def _explode(_c: SensorContext) -> VerificationResult:
        raise RuntimeError("kaboom")

    sensor = Sensor(
        config=VerificationSensorConfig(name="boom", kind="computational", phases=("after_task",)),
        fn=_explode,
    )
    result = await sensor.invoke(_ctx())
    assert result.passed is False
    assert "kaboom" in result.message


class _Person(BaseModel):
    name: str
    age: int


async def test_schema_sensor_passes_for_valid_json() -> None:
    sensor = schema_sensor(_Person)
    result = await sensor.invoke(_ctx(output='{"name": "Ada", "age": 30}'))
    assert result.passed is True


async def test_schema_sensor_fails_for_invalid_json() -> None:
    sensor = schema_sensor(_Person)
    result = await sensor.invoke(_ctx(output="not json"))
    assert result.passed is False
    assert "valid JSON" in result.message


async def test_schema_sensor_fails_for_schema_violation() -> None:
    sensor = schema_sensor(_Person)
    result = await sensor.invoke(_ctx(output='{"name": "Ada"}'))
    assert result.passed is False
    assert result.feedback_for_agent is not None


def test_verification_result_is_immutable() -> None:
    result = VerificationResult(sensor_name="a", kind="computational", passed=True, severity="info", message="ok")
    with pytest.raises(Exception):
        result.passed = False  # type: ignore[misc]
