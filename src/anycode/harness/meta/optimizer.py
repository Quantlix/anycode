"""Meta-evaluation across blueprints and held-out suites."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass

from anycode.harness.evolution.loop import EvolutionLoop, EvolutionLoopReport
from anycode.harness.meta.blueprint import validate_safety_floors
from anycode.types import EvalReport, EvolutionBlueprint, MetaHarnessReport

LoopFactory = Callable[
    [EvolutionBlueprint, EvalReport, str],
    EvolutionLoop | Awaitable[EvolutionLoop],
]


@dataclass(frozen=True)
class SearchSpace:
    """Set of train and held-out suites used to score a blueprint."""

    train_baselines: tuple[EvalReport, ...]
    heldout_baselines: tuple[EvalReport, ...]

    def __post_init__(self) -> None:
        if not self.train_baselines:
            raise ValueError("SearchSpace must contain at least one train suite")


def _score(report: EvolutionLoopReport) -> float:
    """Pass-rate-like score in [0, 1] based on the final accepted state."""

    total = max(report.final_passed + 0, 1)
    accepted = max(len(report.accepted), 0)
    rejected = max(len(report.rejected), 0)
    if accepted + rejected == 0:
        return float(total) / total
    return accepted / max(accepted + rejected, 1)


async def _materialize(loop: EvolutionLoop | Awaitable[EvolutionLoop]) -> EvolutionLoop:
    if hasattr(loop, "__await__"):
        return await loop  # type: ignore[misc]
    return loop  # type: ignore[return-value]


async def run_blueprint(
    blueprint: EvolutionBlueprint,
    *,
    train_suites: Sequence[EvalReport],
    heldout_suites: Sequence[EvalReport],
    loop_factory: LoopFactory,
) -> MetaHarnessReport:
    """Run *blueprint* across train + held-out suites and aggregate the result."""

    validate_safety_floors(blueprint.safety_floors)

    train_scores: list[float] = []
    heldout_scores: list[float] = []
    convergence: list[int] = []
    accepted = 0
    rejected = 0
    total_cost = 0.0
    regression_runs = 0

    for baseline in train_suites:
        loop = await _materialize(loop_factory(blueprint, baseline, "train"))
        report = await loop.run()
        train_scores.append(_score(report))
        accepted += len(report.accepted)
        rejected += len(report.rejected)
        convergence.append(len(report.iterations))
        for iteration in report.iterations:
            total_cost += iteration.candidate_report.total_cost_usd
            if iteration.outcome.regressions:
                regression_runs += 1

    for baseline in heldout_suites:
        loop = await _materialize(loop_factory(blueprint, baseline, "heldout"))
        report = await loop.run()
        heldout_scores.append(_score(report))

    total_runs = max(accepted + rejected, 1)
    regression_rate = regression_runs / total_runs

    return MetaHarnessReport(
        blueprint_id=blueprint.id,
        train_scores=tuple(train_scores),
        heldout_scores=tuple(heldout_scores),
        convergence_iterations=tuple(convergence),
        accepted_changes=accepted,
        rejected_changes=rejected,
        total_cost_usd=total_cost,
        regression_rate=regression_rate,
        notes=blueprint.description,
    )


class MetaOptimizer:
    """Compare two or more blueprints across train + held-out suites."""

    def __init__(
        self,
        *,
        blueprints: Iterable[EvolutionBlueprint],
        loop_factory: LoopFactory,
        search_space: SearchSpace,
    ) -> None:
        materialized = list(blueprints)
        if len(materialized) < 2:
            raise ValueError("MetaOptimizer requires at least two blueprints to compare")
        for blueprint in materialized:
            validate_safety_floors(blueprint.safety_floors)
        self.blueprints = materialized
        self.loop_factory = loop_factory
        self.search_space = search_space

    async def evaluate(self) -> list[MetaHarnessReport]:
        results: list[MetaHarnessReport] = []
        for blueprint in self.blueprints:
            report = await run_blueprint(
                blueprint,
                train_suites=self.search_space.train_baselines,
                heldout_suites=self.search_space.heldout_baselines,
                loop_factory=self.loop_factory,
            )
            results.append(report)
        return results


__all__ = ["MetaOptimizer", "SearchSpace", "run_blueprint"]
