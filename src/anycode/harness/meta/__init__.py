"""Meta-harness optimization (experimental).

The package treats the evolution loop itself as a versioned blueprint and
evaluates blueprints on train + held-out suites. It is research-grade tooling:
production runtime never imports it. Safety floors declared on each
:class:`EvolutionBlueprint` cannot be relaxed by the optimizer.
"""

from anycode.harness.meta.blueprint import (
    SAFETY_FLOOR_FROZEN_FIELDS,
    BlueprintRegistry,
    build_blueprint,
    validate_safety_floors,
)
from anycode.harness.meta.optimizer import (
    MetaOptimizer,
    SearchSpace,
    run_blueprint,
)
from anycode.harness.meta.report import (
    compare_blueprints,
    render_meta_report,
    save_meta_report,
)

__all__ = [
    "BlueprintRegistry",
    "MetaOptimizer",
    "SAFETY_FLOOR_FROZEN_FIELDS",
    "SearchSpace",
    "build_blueprint",
    "compare_blueprints",
    "render_meta_report",
    "run_blueprint",
    "save_meta_report",
    "validate_safety_floors",
]
