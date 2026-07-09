"""Blueprint construction and safety-floor enforcement."""

from __future__ import annotations

from typing import Final

from anycode.types import AcceptanceThresholds, EvolutionBlueprint

# Fields on AcceptanceThresholds that the meta-optimizer can never relax.
SAFETY_FLOOR_FROZEN_FIELDS: Final[tuple[str, ...]] = ("block_on_safety_regression",)

_HARD_SAFETY_DEFAULTS: Final[dict[str, bool]] = {"block_on_safety_regression": True}


def validate_safety_floors(thresholds: AcceptanceThresholds) -> AcceptanceThresholds:
    """Raise ``ValueError`` if any frozen safety floor has been relaxed.

    Returns the thresholds unmodified on success so callers can chain validation.
    """

    for field in SAFETY_FLOOR_FROZEN_FIELDS:
        if not getattr(thresholds, field, _HARD_SAFETY_DEFAULTS[field]):
            raise ValueError(f"safety floor '{field}' may not be disabled by blueprint optimization")
    if thresholds.max_regressions < 0:
        raise ValueError("max_regressions must be >= 0")
    return thresholds


def build_blueprint(
    *,
    id: str,
    worker_seed: str,
    evaluator_prompt_id: str,
    evolution_prompt_id: str,
    evidence_policy_id: str = "default",
    max_iterations: int = 3,
    acceptance_policy_id: str = "default",
    description: str = "",
    safety_floors: AcceptanceThresholds | None = None,
) -> EvolutionBlueprint:
    """Build a validated :class:`EvolutionBlueprint`.

    The provided ``safety_floors`` are validated to ensure that no hard-coded
    floor (see :data:`SAFETY_FLOOR_FROZEN_FIELDS`) has been disabled.
    """

    if not id.strip():
        raise ValueError("blueprint id must be non-empty")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    floors = validate_safety_floors(safety_floors or AcceptanceThresholds())
    return EvolutionBlueprint(
        id=id,
        worker_seed=worker_seed,
        evaluator_prompt_id=evaluator_prompt_id,
        evolution_prompt_id=evolution_prompt_id,
        evidence_policy_id=evidence_policy_id,
        max_iterations=max_iterations,
        acceptance_policy_id=acceptance_policy_id,
        description=description,
        safety_floors=floors,
    )


class BlueprintRegistry:
    """Versioned, in-memory registry of :class:`EvolutionBlueprint`."""

    def __init__(self) -> None:
        self._blueprints: dict[str, EvolutionBlueprint] = {}

    def register(self, blueprint: EvolutionBlueprint) -> None:
        validate_safety_floors(blueprint.safety_floors)
        if blueprint.id in self._blueprints:
            raise ValueError(f"blueprint '{blueprint.id}' already registered")
        self._blueprints[blueprint.id] = blueprint

    def get(self, blueprint_id: str) -> EvolutionBlueprint | None:
        return self._blueprints.get(blueprint_id)

    def list(self) -> list[EvolutionBlueprint]:
        return list(self._blueprints.values())

    def __len__(self) -> int:
        return len(self._blueprints)


__all__ = [
    "BlueprintRegistry",
    "SAFETY_FLOOR_FROZEN_FIELDS",
    "build_blueprint",
    "validate_safety_floors",
]
