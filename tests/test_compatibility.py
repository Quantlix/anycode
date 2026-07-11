"""Compatibility contracts for public imports and persisted formats."""

from __future__ import annotations

from pathlib import Path

import anycode

_API_BASELINE = Path(__file__).parent / "fixtures" / "compat" / "public_api_v0_6.txt"


def _public_api_baseline() -> set[str]:
    return {line for line in _API_BASELINE.read_text(encoding="utf-8").splitlines() if line}


def test_public_api_declaration_has_no_duplicates() -> None:
    assert len(anycode.__all__) == len(set(anycode.__all__))


def test_v0_6_public_api_remains_importable() -> None:
    baseline = _public_api_baseline()
    declared = set(anycode.__all__)

    assert baseline <= declared, f"Removed public API symbols: {sorted(baseline - declared)}"
    missing_attributes = sorted(name for name in baseline if not hasattr(anycode, name))
    assert not missing_attributes, f"Public API symbols are not importable: {missing_attributes}"
