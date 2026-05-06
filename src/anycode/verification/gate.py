"""Quality gate composition.

A QualityGate runs a collection of sensors at a given phase and turns the
collected VerificationResults into a single QualityGateDecision.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

from anycode.types import (
    GateOutcome,
    QualityGateDecision,
    SensorPhase,
    VerificationResult,
)
from anycode.verification.sensor import Sensor, SensorContext


def decide_gate(
    results: Iterable[VerificationResult],
    *,
    block_on_critical: bool = True,
    retry_on_error: bool = True,
    escalate_on_repeated_block: bool = False,
    repeated_block_count: int = 0,
    repeated_block_threshold: int = 2,
) -> QualityGateDecision:
    """Reduce sensor results into a single gate decision."""
    materialized = tuple(results)
    if not materialized:
        return QualityGateDecision(outcome="pass", results=(), message="no sensors registered")

    has_critical = any(r.severity == "critical" and not r.passed for r in materialized)
    has_error = any(r.severity == "error" and not r.passed for r in materialized)
    has_warning = any(r.severity == "warning" and not r.passed for r in materialized)
    all_passed = all(r.passed for r in materialized)

    outcome: GateOutcome
    if escalate_on_repeated_block and repeated_block_count >= repeated_block_threshold:
        outcome = "escalate"
        message = f"escalating after {repeated_block_count} repeated blocks"
    elif has_critical and block_on_critical:
        outcome = "block"
        message = "critical sensor failure; blocking"
    elif has_error and retry_on_error:
        outcome = "retry"
        message = "error-level failure; retry with feedback"
    elif has_warning:
        outcome = "warn"
        message = "warning-level findings; continue with attached warning"
    elif all_passed:
        outcome = "pass"
        message = "all sensors passed"
    else:
        outcome = "warn"
        message = "non-blocking failures present"

    return QualityGateDecision(outcome=outcome, results=materialized, message=message)


class QualityGate:
    """A reusable composition of sensors keyed by lifecycle phase."""

    def __init__(
        self,
        sensors: Iterable[Sensor],
        *,
        block_on_critical: bool = True,
        retry_on_error: bool = True,
    ) -> None:
        self._sensors = tuple(sensors)
        self._block_on_critical = block_on_critical
        self._retry_on_error = retry_on_error
        self._block_history: int = 0

    @property
    def sensors(self) -> tuple[Sensor, ...]:
        return self._sensors

    def _sensors_for_phase(self, phase: SensorPhase) -> tuple[Sensor, ...]:
        return tuple(s for s in self._sensors if phase in s.phases)

    async def evaluate(self, ctx: SensorContext) -> QualityGateDecision:
        """Run all sensors registered for `ctx.phase` and return a decision."""
        active = self._sensors_for_phase(ctx.phase)
        if not active:
            return QualityGateDecision(outcome="pass", results=(), message=f"no sensors for phase '{ctx.phase}'")
        results = await asyncio.gather(*(s.invoke(ctx) for s in active))
        decision = decide_gate(
            results,
            block_on_critical=self._block_on_critical,
            retry_on_error=self._retry_on_error,
            escalate_on_repeated_block=True,
            repeated_block_count=self._block_history,
        )
        if decision.outcome == "block":
            self._block_history += 1
        elif decision.outcome in ("pass", "warn"):
            self._block_history = 0
        return decision
