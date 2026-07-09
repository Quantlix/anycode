"""Tests for the controlled harness evolution loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from anycode.harness.component import make_component
from anycode.harness.evolution import (
    AcceptanceDecision,
    EvolutionLoop,
    EvolutionPolicy,
    acceptance_thresholds,
    apply_thresholds,
    build_change_manifest,
    diff_payloads,
    isolated_workspace,
    load_change_manifest,
    materialize_workspace,
    propose_change,
    save_change_manifest,
)
from anycode.harness.evolution.workspace import _safe_filename
from anycode.types import (
    AcceptanceThresholds,
    EvalReport,
    EvalScenarioResult,
    EvidencePacket,
    FailureCategory,
    HarnessChangePrediction,
)


def _report(*, variant: str, results: tuple[EvalScenarioResult, ...], runtime: float = 1.0) -> EvalReport:
    passed = sum(1 for r in results if r.passed)
    return EvalReport(
        suite_name="suite",
        harness_variant=variant,
        total_scenarios=len(results),
        passed=passed,
        failed=len(results) - passed,
        total_runtime_seconds=runtime,
        total_input_tokens=0,
        total_output_tokens=0,
        total_cost_usd=0.0,
        scenario_results=results,
    )


def _scenario(name: str, *, passed: bool) -> EvalScenarioResult:
    return EvalScenarioResult(
        scenario_name=name,
        passed=passed,
        output="ok" if passed else "no",
        runtime_seconds=0.01,
        turns=1,
        tool_calls=0,
    )


def _editable_component(component_id: str = "prompt:agent:alice") -> Any:
    return make_component(
        id=component_id,
        kind="prompt",
        source="config",
        owner="config",
        description="agent prompt",
        payload={"text": "v1"},
    )


# -- change manifest --


def test_build_change_manifest_validates_inputs() -> None:
    prediction = HarnessChangePrediction(
        metric="passed",
        expected_direction="increase",
        rationale="adding fewer instructions should help",
    )
    manifest = build_change_manifest(
        component_ids=("prompt:agent:alice",),
        summary="trim system prompt",
        predictions=[prediction],
        rollback_plan="restore previous prompt",
    )
    assert manifest.summary == "trim system prompt"
    assert manifest.predictions[0].metric == "passed"
    assert manifest.rollback_plan == "restore previous prompt"


def test_build_change_manifest_rejects_missing_predictions() -> None:
    with pytest.raises(ValueError):
        build_change_manifest(
            component_ids=("x",),
            summary="x",
            predictions=[],
            rollback_plan="x",
        )


def test_build_change_manifest_rejects_empty_component_ids() -> None:
    prediction = HarnessChangePrediction(metric="passed", expected_direction="increase", rationale="r")
    with pytest.raises(ValueError):
        build_change_manifest(
            component_ids=(),
            summary="x",
            predictions=[prediction],
            rollback_plan="x",
        )


def test_build_change_manifest_rejects_empty_summary_or_rollback() -> None:
    prediction = HarnessChangePrediction(metric="passed", expected_direction="increase", rationale="r")
    with pytest.raises(ValueError):
        build_change_manifest(
            component_ids=("x",),
            summary="   ",
            predictions=[prediction],
            rollback_plan="ok",
        )
    with pytest.raises(ValueError):
        build_change_manifest(
            component_ids=("x",),
            summary="x",
            predictions=[prediction],
            rollback_plan="",
        )


def test_diff_payloads_produces_unified_diff() -> None:
    edit = diff_payloads(
        component_id="prompt:agent:alice",
        before={"text": "hello"},
        after={"text": "hi"},
    )
    assert edit.before_checksum != edit.after_checksum
    assert "@before" in edit.diff and "@after" in edit.diff


def test_save_and_load_change_manifest(tmp_path: Path) -> None:
    prediction = HarnessChangePrediction(metric="passed", expected_direction="increase", rationale="r")
    manifest = build_change_manifest(
        component_ids=("a",),
        summary="s",
        predictions=[prediction],
        rollback_plan="rp",
    )
    target = save_change_manifest(manifest, tmp_path / "manifest.json")
    restored = load_change_manifest(target)
    assert restored.id == manifest.id


# -- workspace --


def test_materialize_workspace_writes_json_per_component(tmp_path: Path) -> None:
    ws = materialize_workspace({"prompt:a": {"text": "hi"}}, root=tmp_path / "ws")
    assert ws.root.exists()
    files = list(ws.root.glob("*.json"))
    assert files


def test_workspace_read_write_and_rollback(tmp_path: Path) -> None:
    ws = materialize_workspace({"prompt:a": {"text": "v1"}}, root=tmp_path / "ws")
    before = ws.read("prompt:a")
    ws.write("prompt:a", {"text": "v2"})
    assert ws.read("prompt:a") == {"text": "v2"}
    ws.rollback("prompt:a", before)
    assert ws.read("prompt:a") == before


def test_workspace_read_unknown_component_raises(tmp_path: Path) -> None:
    ws = materialize_workspace({}, root=tmp_path / "ws")
    with pytest.raises(KeyError):
        ws.read("missing")


def test_isolated_workspace_cleans_up_on_exit() -> None:
    with isolated_workspace({"prompt:a": {"text": "hi"}}) as ws:
        path = ws.root
        assert path.exists()
    assert not path.exists()


def test_safe_filename_normalizes_unusual_ids() -> None:
    assert _safe_filename("a/b\\c") == "a_b_c"
    assert _safe_filename("") == "component"


# -- propose_change --


def test_propose_change_rejects_non_editable() -> None:
    component = make_component(
        id="tool:read",
        kind="tool",
        source="x",
        owner="core",
        description="d",
        payload={"x": 1},
    )
    with pytest.raises(ValueError):
        propose_change(
            component=component,
            before={"x": 1},
            after={"x": 2},
            summary="s",
            predictions=[HarnessChangePrediction(metric="passed", expected_direction="increase", rationale="r")],
        )


def test_propose_change_emits_single_edit_manifest() -> None:
    component = _editable_component()
    manifest = propose_change(
        component=component,
        before={"text": "v1"},
        after={"text": "v2"},
        summary="bump",
        predictions=[HarnessChangePrediction(metric="passed", expected_direction="increase", rationale="r")],
    )
    assert manifest.component_ids == (component.id,)
    assert len(manifest.edits) == 1
    assert manifest.safety_review_required is True


# -- acceptance --


def test_apply_thresholds_accepts_pure_improvement() -> None:
    baseline = _report(variant="b", results=(_scenario("a", passed=False), _scenario("b", passed=True)))
    candidate = _report(variant="c", results=(_scenario("a", passed=True), _scenario("b", passed=True)))
    prediction = HarnessChangePrediction(metric="passed", expected_direction="increase", rationale="r")
    manifest = build_change_manifest(component_ids=("c",), summary="s", predictions=[prediction], rollback_plan="rb")
    decision = apply_thresholds(manifest=manifest, baseline=baseline, candidate=candidate)
    assert isinstance(decision, AcceptanceDecision)
    assert decision.accepted is True
    assert decision.outcome.candidate_passed == 2
    assert decision.outcome.improvements


def test_apply_thresholds_rejects_regression() -> None:
    baseline = _report(variant="b", results=(_scenario("a", passed=True), _scenario("b", passed=True)))
    candidate = _report(variant="c", results=(_scenario("a", passed=False), _scenario("b", passed=True)))
    prediction = HarnessChangePrediction(metric="passed", expected_direction="increase", rationale="r")
    manifest = build_change_manifest(component_ids=("c",), summary="s", predictions=[prediction], rollback_plan="rb")
    decision = apply_thresholds(manifest=manifest, baseline=baseline, candidate=candidate)
    assert decision.accepted is False
    assert decision.outcome.regressions


def test_apply_thresholds_blocks_safety_regression() -> None:
    base_results = (_scenario("a", passed=True),)
    cand_results = (_scenario("a", passed=True),)
    baseline = _report(variant="b", results=base_results)
    candidate = EvalReport(
        suite_name="suite",
        harness_variant="c",
        total_scenarios=1,
        passed=1,
        failed=0,
        total_runtime_seconds=1.0,
        total_input_tokens=0,
        total_output_tokens=0,
        total_cost_usd=0.0,
        total_verification_failures=2,
        scenario_results=cand_results,
    )
    prediction = HarnessChangePrediction(metric="passed", expected_direction="increase", rationale="r")
    manifest = build_change_manifest(component_ids=("c",), summary="s", predictions=[prediction], rollback_plan="rb")
    decision = apply_thresholds(manifest=manifest, baseline=baseline, candidate=candidate)
    assert decision.accepted is False
    assert any("safety" in reason.lower() for reason in decision.outcome.reasons)


def test_apply_thresholds_runtime_cap() -> None:
    baseline = _report(variant="b", results=(_scenario("a", passed=True),), runtime=1.0)
    candidate = _report(variant="c", results=(_scenario("a", passed=True),), runtime=10.0)
    prediction = HarnessChangePrediction(metric="passed", expected_direction="unchanged", rationale="r")
    manifest = build_change_manifest(component_ids=("c",), summary="s", predictions=[prediction], rollback_plan="rb")
    decision = apply_thresholds(
        manifest=manifest,
        baseline=baseline,
        candidate=candidate,
        thresholds=acceptance_thresholds(max_runtime_delta_seconds=2.0),
    )
    assert decision.accepted is False


def test_apply_thresholds_requires_at_least_one_confirmed_prediction() -> None:
    baseline = _report(variant="b", results=(_scenario("a", passed=True),))
    candidate = _report(variant="c", results=(_scenario("a", passed=True),))
    prediction = HarnessChangePrediction(metric="passed", expected_direction="increase", rationale="r")
    manifest = build_change_manifest(component_ids=("c",), summary="s", predictions=[prediction], rollback_plan="rb")
    decision = apply_thresholds(manifest=manifest, baseline=baseline, candidate=candidate)
    # No improvement in passed: direction was unchanged, prediction was increase → rejected
    assert decision.accepted is False
    assert any("prediction" in reason.lower() for reason in decision.outcome.reasons)


# -- evolution loop --


def _build_loop_inputs():
    component = _editable_component()
    baseline = _report(variant="baseline", results=(_scenario("a", passed=False),))
    payloads = {component.id: {"text": "v1"}}
    return component, baseline, payloads


@pytest.mark.asyncio
async def test_evolution_loop_accepts_when_candidate_improves(tmp_path: Path) -> None:
    component, baseline, payloads = _build_loop_inputs()

    def evolve_fn(_comp, payload):
        return {**payload, "text": payload["text"] + "+"}

    async def worker_fn(_ws):
        return _report(variant="cand", results=(_scenario("a", passed=True),))

    loop = EvolutionLoop(
        baseline_report=baseline,
        baseline_payloads=payloads,
        components={component.id: component},
        evolve_fn=evolve_fn,
        worker_fn=worker_fn,
        policy=EvolutionPolicy(max_iterations=1, patch_dir=tmp_path / "patches"),
    )
    report = await loop.run()
    assert report.accepted
    assert not report.rejected
    assert report.iterations[0].outcome.status == "accepted"
    assert report.iterations[0].outcome.patch_path is not None
    assert Path(report.iterations[0].outcome.patch_path).exists()


@pytest.mark.asyncio
async def test_evolution_loop_rolls_back_when_candidate_regresses(tmp_path: Path) -> None:
    component, baseline, payloads = _build_loop_inputs()
    # baseline has 0 passes → candidate has 0 passes; predicted increase → rejected.

    def evolve_fn(_comp, payload):
        return {**payload, "text": payload["text"] + "-"}

    async def worker_fn(_ws):
        return _report(variant="cand", results=(_scenario("a", passed=False),))

    loop = EvolutionLoop(
        baseline_report=baseline,
        baseline_payloads=payloads,
        components={component.id: component},
        evolve_fn=evolve_fn,
        worker_fn=worker_fn,
        policy=EvolutionPolicy(max_iterations=1),
    )
    report = await loop.run()
    assert report.rejected
    assert report.iterations[0].outcome.status == "rolled_back"


@pytest.mark.asyncio
async def test_evolution_loop_terminates_on_no_op() -> None:
    component, baseline, payloads = _build_loop_inputs()

    def evolve_fn(_comp, payload):
        return payload  # no-op

    async def worker_fn(_ws):
        return baseline

    loop = EvolutionLoop(
        baseline_report=baseline,
        baseline_payloads=payloads,
        components={component.id: component},
        evolve_fn=evolve_fn,
        worker_fn=worker_fn,
        policy=EvolutionPolicy(max_iterations=3),
    )
    report = await loop.run()
    assert not report.iterations  # terminated immediately


@pytest.mark.asyncio
async def test_evolution_loop_raises_when_no_editable_components() -> None:
    baseline = _report(variant="b", results=(_scenario("a", passed=True),))
    component = make_component(
        id="tool:locked",
        kind="tool",
        source="x",
        owner="core",
        description="",
        payload={},
    )
    payloads = {component.id: {"x": 1}}

    async def worker_fn(_ws):
        return baseline

    def evolve_fn(_comp, payload):
        return {**payload, "x": 2}

    loop = EvolutionLoop(
        baseline_report=baseline,
        baseline_payloads=payloads,
        components={component.id: component},
        evolve_fn=evolve_fn,
        worker_fn=worker_fn,
        policy=EvolutionPolicy(max_iterations=1),
    )
    import pytest as _pytest

    with _pytest.raises(RuntimeError):
        await loop.step(iteration=0, workspace=materialize_workspace(payloads))


@pytest.mark.asyncio
async def test_evolution_loop_supports_sync_worker(tmp_path: Path) -> None:
    component, baseline, payloads = _build_loop_inputs()

    def evolve_fn(_comp, payload):
        return {**payload, "text": "v2"}

    def worker_fn(_ws):
        return _report(variant="cand", results=(_scenario("a", passed=True),))

    loop = EvolutionLoop(
        baseline_report=baseline,
        baseline_payloads=payloads,
        components={component.id: component},
        evolve_fn=evolve_fn,
        worker_fn=worker_fn,
        policy=EvolutionPolicy(max_iterations=1),
    )
    report = await loop.run()
    assert report.accepted


@pytest.mark.asyncio
async def test_evolution_loop_uses_evidence_packets(tmp_path: Path) -> None:
    component, baseline, payloads = _build_loop_inputs()
    packet = EvidencePacket(id="p-1", category=FailureCategory.TOOL_RUNTIME_ERROR, summary="boom")

    def evolve_fn(_comp, payload):
        return {**payload, "text": "v2"}

    async def worker_fn(_ws):
        return _report(variant="cand", results=(_scenario("a", passed=True),))

    loop = EvolutionLoop(
        baseline_report=baseline,
        baseline_payloads=payloads,
        components={component.id: component},
        evolve_fn=evolve_fn,
        worker_fn=worker_fn,
        evidence=[packet],
        policy=EvolutionPolicy(max_iterations=1),
    )
    report = await loop.run()
    assert report.iterations[0].manifest.evidence_packet_ids == ("p-1",)


def test_acceptance_thresholds_immutable() -> None:
    thresholds = AcceptanceThresholds(max_regressions=0)
    with pytest.raises(Exception):
        thresholds.max_regressions = 5  # type: ignore[misc]
