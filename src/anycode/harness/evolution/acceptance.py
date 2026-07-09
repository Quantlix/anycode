"""Acceptance policy and threshold enforcement.

Acceptance is intentionally conservative:

- Critical safety regressions block acceptance unconditionally.
- Pass-count regressions, runtime regressions, and cost regressions can be
  capped via :class:`AcceptanceThresholds`.
- Predictions are compared against measured outcomes — a change whose every
  prediction was falsified is rejected even when it appears to "improve" some
  unrelated metric.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from anycode.eval.report import compare_reports
from anycode.types import (
    AcceptanceThresholds,
    EvalReport,
    HarnessChangeManifest,
    HarnessChangeOutcome,
    HarnessChangePrediction,
)

_METRIC_RESOLVERS: dict[str, Callable[[EvalReport], float]] = {
    "passed": lambda report: float(report.passed),
    "failed": lambda report: float(report.failed),
    "total_runtime_seconds": lambda report: float(report.total_runtime_seconds),
    "total_cost_usd": lambda report: float(report.total_cost_usd),
    "total_retries": lambda report: float(report.total_retries),
    "total_verification_failures": lambda report: float(report.total_verification_failures),
}


@dataclass(frozen=True)
class AcceptanceDecision:
    accepted: bool
    outcome: HarnessChangeOutcome


def acceptance_thresholds(**overrides: object) -> AcceptanceThresholds:
    """Return a :class:`AcceptanceThresholds` with optional overrides."""

    return AcceptanceThresholds(**overrides)  # type: ignore[arg-type]


def _metric_value(report: EvalReport, metric: str) -> float | None:
    resolver = _METRIC_RESOLVERS.get(metric)
    return resolver(report) if resolver else None


def _evaluate_prediction(
    prediction: HarnessChangePrediction,
    baseline: EvalReport,
    candidate: EvalReport,
) -> dict[str, str | float | bool | None]:
    base_value = _metric_value(baseline, prediction.metric)
    cand_value = _metric_value(candidate, prediction.metric)
    measured_delta: float | None
    if base_value is None or cand_value is None:
        measured_delta = None
        actual_direction = "unknown"
    else:
        measured_delta = cand_value - base_value
        if abs(measured_delta) < 1e-9:
            actual_direction = "unchanged"
        elif measured_delta > 0:
            actual_direction = "increase"
        else:
            actual_direction = "decrease"
    confirmed = actual_direction == prediction.expected_direction if actual_direction != "unknown" else False
    return {
        "metric": prediction.metric,
        "expected": prediction.expected_direction,
        "actual": actual_direction,
        "expected_delta": prediction.expected_delta,
        "measured_delta": measured_delta,
        "confirmed": confirmed,
    }


def apply_thresholds(
    *,
    manifest: HarnessChangeManifest,
    baseline: EvalReport,
    candidate: EvalReport,
    thresholds: AcceptanceThresholds | None = None,
) -> AcceptanceDecision:
    """Compare baseline + candidate reports against the manifest's predictions."""

    thresholds = thresholds or AcceptanceThresholds()
    diff = compare_reports(baseline, candidate)
    regressions = tuple(diff["regressions"])
    improvements = tuple(diff["improvements"])

    predicted_vs_measured = tuple(_evaluate_prediction(p, baseline, candidate) for p in manifest.predictions)

    reasons: list[str] = []
    accepted = True

    pass_delta = candidate.passed - baseline.passed
    if pass_delta < thresholds.min_pass_delta:
        accepted = False
        reasons.append(f"pass-count delta {pass_delta} is below floor {thresholds.min_pass_delta}")

    if len(regressions) > thresholds.max_regressions:
        accepted = False
        reasons.append(f"{len(regressions)} regressions exceeds max_regressions={thresholds.max_regressions}")

    if thresholds.max_runtime_delta_seconds is not None:
        runtime_delta = candidate.total_runtime_seconds - baseline.total_runtime_seconds
        if runtime_delta > thresholds.max_runtime_delta_seconds:
            accepted = False
            reasons.append(f"runtime delta {runtime_delta:.2f}s exceeds {thresholds.max_runtime_delta_seconds}s")

    if thresholds.max_cost_delta_usd is not None:
        cost_delta = candidate.total_cost_usd - baseline.total_cost_usd
        if cost_delta > thresholds.max_cost_delta_usd:
            accepted = False
            reasons.append(f"cost delta ${cost_delta:.4f} exceeds ${thresholds.max_cost_delta_usd:.4f}")

    if thresholds.block_on_safety_regression:
        baseline_safety = baseline.total_verification_failures
        candidate_safety = candidate.total_verification_failures
        if candidate_safety > baseline_safety:
            accepted = False
            reasons.append(f"safety regression: verification failures {baseline_safety}→{candidate_safety}")

    if predicted_vs_measured and not any(item["confirmed"] for item in predicted_vs_measured):
        accepted = False
        reasons.append("no prediction was confirmed by measurement")

    status = "accepted" if accepted else "rejected"
    outcome = HarnessChangeOutcome(
        manifest_id=manifest.id,
        status=status,  # type: ignore[arg-type]
        baseline_passed=baseline.passed,
        candidate_passed=candidate.passed,
        regressions=regressions,
        improvements=improvements,
        predicted_vs_measured=predicted_vs_measured,
        reasons=tuple(reasons) or ("no concerns raised by acceptance policy",),
    )
    return AcceptanceDecision(accepted=accepted, outcome=outcome)


def summarize_predictions(
    predictions: Iterable[HarnessChangePrediction],
) -> str:
    """Render predictions as a one-line, human-readable string."""

    parts: list[str] = []
    for prediction in predictions:
        delta = f"{prediction.expected_delta:+.4f}" if prediction.expected_delta is not None else "?"
        parts.append(f"{prediction.metric} {prediction.expected_direction} (delta={delta})")
    return "; ".join(parts) if parts else "no predictions"


__all__ = [
    "AcceptanceDecision",
    "acceptance_thresholds",
    "apply_thresholds",
    "summarize_predictions",
]
