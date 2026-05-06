"""EvalScenario loading from YAML/JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anycode.types import EvalScenario


def load_scenario(path: str | Path) -> EvalScenario:
    """Load a single scenario from a YAML or JSON file."""
    raw = Path(path).read_text(encoding="utf-8")
    data = _parse_payload(raw, path)
    return EvalScenario.model_validate(data)


def load_scenarios(path: str | Path) -> list[EvalScenario]:
    """Load a suite definition: either a list of scenarios or a dict with `scenarios:` key."""
    raw = Path(path).read_text(encoding="utf-8")
    data = _parse_payload(raw, path)
    if isinstance(data, dict) and "scenarios" in data:
        items = data["scenarios"]
    elif isinstance(data, list):
        items = data
    else:
        items = [data]
    return [EvalScenario.model_validate(item) for item in items]


def _parse_payload(raw: str, source: str | Path) -> Any:
    suffix = Path(source).suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("YAML scenarios require PyYAML. Install with: pip install pyyaml") from exc
        return yaml.safe_load(raw)
    return json.loads(raw)
