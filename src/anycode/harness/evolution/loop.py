"""Controlled harness evolution loop.

This module wires together four loop roles:

- **Worker** — executes the configured eval suite under a candidate harness.
- **Evaluator** — compares the resulting :class:`EvalReport` against a
  baseline using :func:`anycode.harness.evolution.acceptance.apply_thresholds`.
- **Evolution agent** — proposes edits to editable harness components.
- **Gatekeeper** — enforces dry-run defaults, safety floors, and emits patches.

The default :class:`EvolutionPolicy` is dry-run only: edits are recorded as
patch artifacts and *never* written back to the user's repository. The
``allow_filesystem_writes`` flag is off by default and is intended only for
experimental/CI tooling.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from anycode.harness.evolution.acceptance import (
    AcceptanceDecision,
    apply_thresholds,
)
from anycode.harness.evolution.manifest import (
    build_change_manifest,
    diff_payloads,
    save_change_manifest,
)
from anycode.harness.evolution.workspace import (
    HarnessWorkspace,
    materialize_workspace,
)
from anycode.types import (
    AcceptanceThresholds,
    EvalReport,
    EvidencePacket,
    HarnessChangeEdit,
    HarnessChangeManifest,
    HarnessChangeOutcome,
    HarnessChangePrediction,
    HarnessComponent,
)

EvolveFn = Callable[[HarnessComponent, dict[str, Any]], dict[str, Any]]
WorkerFn = Callable[[HarnessWorkspace], Awaitable[EvalReport] | EvalReport]


@dataclass(frozen=True)
class EvolutionPolicy:
    """Governance settings for a single :class:`EvolutionLoop` invocation."""

    max_iterations: int = 3
    dry_run: bool = True
    allow_filesystem_writes: bool = False
    thresholds: AcceptanceThresholds = field(default_factory=AcceptanceThresholds)
    patch_dir: Path | None = None


@dataclass(frozen=True)
class EvolutionResult:
    """Outcome of a single evolution iteration."""

    iteration: int
    manifest: HarnessChangeManifest
    outcome: HarnessChangeOutcome
    decision: AcceptanceDecision
    candidate_report: EvalReport


@dataclass(frozen=True)
class EvolutionLoopReport:
    """Aggregate report across every iteration of an :class:`EvolutionLoop`."""

    iterations: tuple[EvolutionResult, ...]
    accepted: tuple[str, ...]
    rejected: tuple[str, ...]
    baseline_passed: int
    final_passed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": list(self.accepted),
            "rejected": list(self.rejected),
            "baseline_passed": self.baseline_passed,
            "final_passed": self.final_passed,
            "iterations": [
                {
                    "iteration": item.iteration,
                    "manifest_id": item.manifest.id,
                    "outcome": item.outcome.model_dump(),
                }
                for item in self.iterations
            ],
        }


def propose_change(
    *,
    component: HarnessComponent,
    before: dict[str, Any],
    after: dict[str, Any],
    summary: str,
    predictions: Iterable[HarnessChangePrediction],
    rollback_plan: str | None = None,
    evidence_packets: Iterable[EvidencePacket] = (),
    safety_review_required: bool = True,
    note: str = "",
) -> HarnessChangeManifest:
    """Helper that wraps :func:`build_change_manifest` for a single-edit change."""

    if not component.editable:
        raise ValueError(f"component '{component.id}' is not editable")
    edit: HarnessChangeEdit = diff_payloads(
        component_id=component.id,
        before=before,
        after=after,
        note=note,
    )
    rollback = rollback_plan or f"restore component '{component.id}' to checksum {edit.before_checksum}"
    return build_change_manifest(
        component_ids=(component.id,),
        summary=summary,
        predictions=list(predictions),
        rollback_plan=rollback,
        edits=(edit,),
        evidence_packets=evidence_packets,
        safety_review_required=safety_review_required,
    )


class EvolutionLoop:
    """Coordinate proposal → application → evaluation → acceptance."""

    def __init__(
        self,
        *,
        baseline_report: EvalReport,
        baseline_payloads: dict[str, dict[str, Any]],
        components: dict[str, HarnessComponent],
        evolve_fn: EvolveFn,
        worker_fn: WorkerFn,
        policy: EvolutionPolicy | None = None,
        evidence: Iterable[EvidencePacket] = (),
        proposer: Callable[..., HarnessChangeManifest] | None = None,
    ) -> None:
        self.baseline_report = baseline_report
        self.baseline_payloads = baseline_payloads
        self.components = components
        self.evolve_fn = evolve_fn
        self.worker_fn = worker_fn
        self.policy = policy or EvolutionPolicy()
        self.evidence = tuple(evidence)
        self._proposer = proposer or propose_change

    async def _run_worker(self, workspace: HarnessWorkspace) -> EvalReport:
        outcome = self.worker_fn(workspace)
        if hasattr(outcome, "__await__"):
            return await outcome  # type: ignore[misc]
        return outcome  # type: ignore[return-value]

    def _select_editable(self) -> list[HarnessComponent]:
        return [component for component in self.components.values() if component.editable]

    def _ensure_patch_dir(self) -> Path | None:
        if self.policy.patch_dir is None:
            return None
        path = Path(self.policy.patch_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _emit_patch(self, manifest: HarnessChangeManifest, outcome: HarnessChangeOutcome) -> str | None:
        patch_dir = self._ensure_patch_dir()
        if patch_dir is None:
            return None
        path = patch_dir / f"{manifest.id}.patch.json"
        payload = {
            "manifest": manifest.model_dump(),
            "outcome": outcome.model_dump(),
            "emitted_at": datetime.now(UTC).isoformat(),
            "policy": {
                "dry_run": self.policy.dry_run,
                "max_iterations": self.policy.max_iterations,
            },
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return str(path)

    async def step(self, *, iteration: int, workspace: HarnessWorkspace) -> EvolutionResult:
        """Run one proposal → apply → evaluate → accept cycle."""

        editable = self._select_editable()
        if not editable:
            raise RuntimeError("no editable components are registered")
        target = editable[iteration % len(editable)]
        before = workspace.read(target.id)
        proposed = self.evolve_fn(target, before)
        if not isinstance(proposed, dict):
            raise TypeError(f"evolve_fn must return a dict; got {type(proposed).__name__}")
        if proposed == before:
            raise ValueError(f"evolve_fn produced no-op edit for '{target.id}'")
        workspace.write(target.id, proposed)

        prediction = HarnessChangePrediction(
            metric="passed",
            expected_direction="increase",
            expected_delta=None,
            rationale=f"iteration {iteration} edit on '{target.id}' should not lose passing scenarios",
        )
        manifest = self._proposer(
            component=target,
            before=before,
            after=proposed,
            summary=f"iteration {iteration}: edit {target.id}",
            predictions=[prediction],
            rollback_plan=f"restore '{target.id}' to baseline payload",
            evidence_packets=self.evidence,
            safety_review_required=True,
            note=f"iteration={iteration}",
        )

        candidate_report = await self._run_worker(workspace)
        decision = apply_thresholds(
            manifest=manifest,
            baseline=self.baseline_report,
            candidate=candidate_report,
            thresholds=self.policy.thresholds,
        )

        outcome = decision.outcome
        if not decision.accepted:
            workspace.rollback(target.id, before)
            outcome = outcome.model_copy(update={"status": "rolled_back"})
        else:
            outcome = outcome.model_copy(update={"status": "accepted"})

        patch_path = self._emit_patch(manifest, outcome)
        if patch_path is not None:
            outcome = outcome.model_copy(update={"patch_path": patch_path})

        return EvolutionResult(
            iteration=iteration,
            manifest=manifest,
            outcome=outcome,
            decision=AcceptanceDecision(accepted=decision.accepted, outcome=outcome),
            candidate_report=candidate_report,
        )

    async def run(self) -> EvolutionLoopReport:
        """Run up to ``policy.max_iterations`` proposal cycles."""

        workspace = materialize_workspace(self.baseline_payloads)
        results: list[EvolutionResult] = []
        accepted_ids: list[str] = []
        rejected_ids: list[str] = []
        final_passed = self.baseline_report.passed
        last_passed = final_passed

        for iteration in range(self.policy.max_iterations):
            try:
                result = await self.step(iteration=iteration, workspace=workspace)
            except ValueError:
                # evolve_fn produced a no-op; treat as convergence.
                break
            results.append(result)
            if result.outcome.status == "accepted":
                accepted_ids.append(result.manifest.id)
                last_passed = result.candidate_report.passed
                final_passed = last_passed
            else:
                rejected_ids.append(result.manifest.id)
        return EvolutionLoopReport(
            iterations=tuple(results),
            accepted=tuple(accepted_ids),
            rejected=tuple(rejected_ids),
            baseline_passed=self.baseline_report.passed,
            final_passed=final_passed,
        )

    def emit_manifest(self, manifest: HarnessChangeManifest, target: str | Path) -> Path:
        return save_change_manifest(manifest, target)


__all__ = [
    "EvolutionLoop",
    "EvolutionLoopReport",
    "EvolutionPolicy",
    "EvolutionResult",
    "propose_change",
]
