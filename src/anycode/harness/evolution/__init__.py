"""Controlled harness evolution loop.

The package exposes a worker/evaluator/evolution-agent/gatekeeper loop with an
isolated workspace, change manifests with falsifiable predictions, and
automatic rollback when acceptance thresholds are violated.

Production runtime does **not** import this package. It is reserved for
experimental ``anycode harness evolve`` runs and tests.
"""

from anycode.harness.evolution.acceptance import (
    AcceptanceDecision,
    acceptance_thresholds,
    apply_thresholds,
)
from anycode.harness.evolution.loop import (
    EvolutionLoop,
    EvolutionLoopReport,
    EvolutionPolicy,
    EvolutionResult,
    propose_change,
)
from anycode.harness.evolution.manifest import (
    build_change_manifest,
    diff_payloads,
    load_change_manifest,
    save_change_manifest,
)
from anycode.harness.evolution.workspace import (
    HarnessWorkspace,
    isolated_workspace,
    materialize_workspace,
)

__all__ = [
    "AcceptanceDecision",
    "EvolutionLoop",
    "EvolutionLoopReport",
    "EvolutionPolicy",
    "EvolutionResult",
    "HarnessWorkspace",
    "acceptance_thresholds",
    "apply_thresholds",
    "build_change_manifest",
    "diff_payloads",
    "isolated_workspace",
    "load_change_manifest",
    "materialize_workspace",
    "propose_change",
    "save_change_manifest",
]
