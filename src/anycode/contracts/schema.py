"""Deterministic JSON Schema generation for semantic contracts."""

from __future__ import annotations

import json
from pathlib import Path

from anycode.contracts.models import CONTRACT_MODELS, CONTRACT_SCHEMA_VERSION, ContractModel

SCHEMA_DIRECTORY_NAME = "v1"


def schema_filename(model: type[ContractModel]) -> str:
    return f"{model.__name__.lower()}.schema.json"


def render_contract_schema(model: type[ContractModel]) -> str:
    return json.dumps(model.model_json_schema(mode="serialization"), indent=2, sort_keys=True) + "\n"


def contract_schema_bundle() -> dict[str, object]:
    return {
        "contract_version": CONTRACT_SCHEMA_VERSION,
        "models": {model.__name__: model.model_json_schema(mode="serialization") for model in CONTRACT_MODELS},
    }


def synchronize_contract_schemas(output_dir: str | Path, *, check: bool = False) -> list[str]:
    root = Path(output_dir)
    mismatches: list[str] = []
    for model in CONTRACT_MODELS:
        target = root / schema_filename(model)
        expected = render_contract_schema(model)
        actual = target.read_text(encoding="utf-8") if target.is_file() else None
        if actual == expected:
            continue
        mismatches.append(target.name)
        if not check:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(expected, encoding="utf-8")
    return mismatches
