"""Tests for the meta-harness optimization layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from anycode.harness.component import make_component
from anycode.harness.evolution import EvolutionLoop, EvolutionPolicy
from anycode.harness.meta import (
    BlueprintRegistry,
    MetaOptimizer,
    SearchSpace,
    build_blueprint,
    compare_blueprints,
    render_meta_report,
    run_blueprint,
    save_meta_report,
    validate_safety_floors,
)
from anycode.types import (
    AcceptanceThresholds,
    EvalReport,
    EvalScenarioResult,
    EvolutionBlueprint,
)


def _report(passed: bool = True) -> EvalReport:
    result = EvalScenarioResult(
        scenario_name="a",
        passed=passed,
        output="ok" if passed else "no",
        runtime_seconds=0.01,
        turns=1,
        tool_calls=0,
    )
    return EvalReport(
        suite_name="suite",
        harness_variant="baseline",
        total_scenarios=1,
        passed=1 if passed else 0,
        failed=0 if passed else 1,
        total_runtime_seconds=0.01,
        total_input_tokens=0,
        total_output_tokens=0,
        total_cost_usd=0.0,
        scenario_results=(result,),
    )


# -- blueprint --


def test_build_blueprint_validates_iterations() -> None:
    with pytest.raises(ValueError):
        build_blueprint(
            id="b1",
            worker_seed="seed",
            evaluator_prompt_id="e",
            evolution_prompt_id="v",
            max_iterations=0,
        )


def test_build_blueprint_rejects_empty_id() -> None:
    with pytest.raises(ValueError):
        build_blueprint(id="  ", worker_seed="s", evaluator_prompt_id="e", evolution_prompt_id="v")


def test_validate_safety_floors_rejects_disabled_safety() -> None:
    relaxed = AcceptanceThresholds(block_on_safety_regression=False)
    with pytest.raises(ValueError):
        validate_safety_floors(relaxed)


def test_validate_safety_floors_rejects_negative_max_regressions() -> None:
    relaxed = AcceptanceThresholds(max_regressions=-1)
    with pytest.raises(ValueError):
        validate_safety_floors(relaxed)


def test_build_blueprint_with_explicit_safety_floors() -> None:
    floors = AcceptanceThresholds(max_regressions=0)
    blueprint = build_blueprint(
        id="b1",
        worker_seed="s",
        evaluator_prompt_id="e",
        evolution_prompt_id="v",
        safety_floors=floors,
    )
    assert blueprint.safety_floors.max_regressions == 0


# -- registry --


def test_blueprint_registry_register_and_get() -> None:
    registry = BlueprintRegistry()
    blueprint = build_blueprint(id="b1", worker_seed="s", evaluator_prompt_id="e", evolution_prompt_id="v")
    registry.register(blueprint)
    assert registry.get("b1") is blueprint
    assert len(registry) == 1


def test_blueprint_registry_rejects_duplicate() -> None:
    registry = BlueprintRegistry()
    blueprint = build_blueprint(id="dup", worker_seed="s", evaluator_prompt_id="e", evolution_prompt_id="v")
    registry.register(blueprint)
    with pytest.raises(ValueError):
        registry.register(blueprint)


def test_blueprint_registry_rejects_unsafe() -> None:
    registry = BlueprintRegistry()
    unsafe = EvolutionBlueprint(
        id="x",
        worker_seed="s",
        evaluator_prompt_id="e",
        evolution_prompt_id="v",
        evidence_policy_id="d",
        safety_floors=AcceptanceThresholds(block_on_safety_regression=False),
    )
    with pytest.raises(ValueError):
        registry.register(unsafe)


# -- run_blueprint --


def _make_loop_factory():
    component = make_component(id="prompt:a", kind="prompt", source="x", owner="config", description="d", payload={"text": "v1"})
    payloads = {component.id: {"text": "v1"}}

    def factory(blueprint, baseline, _suite):
        def evolve_fn(_comp, payload):
            return {**payload, "text": payload["text"] + "+"}

        def worker_fn(_ws):
            return _report(passed=True)

        return EvolutionLoop(
            baseline_report=baseline,
            baseline_payloads=payloads,
            components={component.id: component},
            evolve_fn=evolve_fn,
            worker_fn=worker_fn,
            policy=EvolutionPolicy(max_iterations=blueprint.max_iterations),
        )

    return factory


async def test_run_blueprint_aggregates_scores() -> None:
    blueprint = build_blueprint(id="bp-1", worker_seed="s", evaluator_prompt_id="e", evolution_prompt_id="v", max_iterations=1)
    factory = _make_loop_factory()
    report = await run_blueprint(
        blueprint,
        train_suites=[_report(passed=False)],
        heldout_suites=[_report(passed=False)],
        loop_factory=factory,
    )
    assert report.blueprint_id == "bp-1"
    assert len(report.train_scores) == 1
    assert len(report.heldout_scores) == 1
    assert report.convergence_iterations == (1,)


async def test_run_blueprint_validates_safety_floors_at_call_time() -> None:
    bad = EvolutionBlueprint(
        id="bad",
        worker_seed="s",
        evaluator_prompt_id="e",
        evolution_prompt_id="v",
        evidence_policy_id="d",
        safety_floors=AcceptanceThresholds(block_on_safety_regression=False),
    )
    with pytest.raises(ValueError):
        await run_blueprint(
            bad,
            train_suites=[_report()],
            heldout_suites=[],
            loop_factory=_make_loop_factory(),
        )


# -- meta-optimizer --


async def test_meta_optimizer_compares_two_blueprints() -> None:
    bp_a = build_blueprint(id="A", worker_seed="s", evaluator_prompt_id="e", evolution_prompt_id="v", max_iterations=1)
    bp_b = build_blueprint(id="B", worker_seed="s", evaluator_prompt_id="e", evolution_prompt_id="v", max_iterations=2)
    optimizer = MetaOptimizer(
        blueprints=[bp_a, bp_b],
        loop_factory=_make_loop_factory(),
        search_space=SearchSpace(
            train_baselines=(_report(passed=False),),
            heldout_baselines=(_report(passed=False),),
        ),
    )
    results = await optimizer.evaluate()
    assert {r.blueprint_id for r in results} == {"A", "B"}


def test_meta_optimizer_requires_two_blueprints() -> None:
    bp = build_blueprint(id="only", worker_seed="s", evaluator_prompt_id="e", evolution_prompt_id="v")
    with pytest.raises(ValueError):
        MetaOptimizer(
            blueprints=[bp],
            loop_factory=_make_loop_factory(),
            search_space=SearchSpace(train_baselines=(_report(),), heldout_baselines=()),
        )


def test_search_space_rejects_empty_train() -> None:
    with pytest.raises(ValueError):
        SearchSpace(train_baselines=(), heldout_baselines=())


# -- report rendering and persistence --


def test_save_and_render_meta_report(tmp_path: Path) -> None:
    from anycode.types import MetaHarnessReport

    report = MetaHarnessReport(
        blueprint_id="bp",
        train_scores=(0.8, 0.9),
        heldout_scores=(0.7,),
        convergence_iterations=(1, 2),
        accepted_changes=2,
        rejected_changes=1,
        total_cost_usd=0.05,
        regression_rate=0.33,
        notes="research",
    )
    text = render_meta_report(report)
    assert "Train score" in text
    assert "Held-out score" in text
    target = save_meta_report(report, tmp_path / "meta.json")
    assert target.exists()


def test_compare_blueprints_ranks_by_heldout_score() -> None:
    from anycode.types import MetaHarnessReport

    a = MetaHarnessReport(blueprint_id="A", train_scores=(0.9,), heldout_scores=(0.5,))
    b = MetaHarnessReport(blueprint_id="B", train_scores=(0.7,), heldout_scores=(0.8,))
    summary = compare_blueprints([a, b])
    assert summary["winner"] == "B"
    assert summary["ranking"][0]["blueprint_id"] == "B"
