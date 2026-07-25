"""Guards for the lazily-resolved public API surface."""

from __future__ import annotations

import subprocess
import sys

import pytest

import anycode
from anycode._lazy import build_export_map, lazy_getattr

# Ceilings for a bare `import anycode`. Generous enough to absorb a slow CI machine,
# tight enough that eagerly importing a subsystem again would trip them.
MAX_IMPORT_SECONDS = 1.0
MAX_IMPORTED_MODULES = 250

HEAVY_DEPENDENCIES = ("chromadb", "redis", "boto3", "opentelemetry", "typer", "mcp", "anthropic", "openai", "yaml", "numpy")


def test_every_exported_name_is_mapped() -> None:
    assert set(anycode.__all__) == set(anycode._EXPORTS)


def test_all_has_no_duplicates() -> None:
    assert len(anycode.__all__) == len(set(anycode.__all__))


def test_every_exported_name_resolves() -> None:
    unresolved = [name for name in anycode.__all__ if getattr(anycode, name, None) is None]
    assert unresolved == []


def test_dir_lists_the_public_surface() -> None:
    assert dir(anycode) == sorted(anycode.__all__)


def test_unknown_attribute_suggests_a_close_match() -> None:
    with pytest.raises(AttributeError, match="Did you mean: Agent"):
        anycode.Agnet  # type: ignore[attr-defined] # noqa: B018


def test_unknown_attribute_without_a_match_points_at_the_api_command() -> None:
    with pytest.raises(AttributeError, match="anycode api"):
        anycode.zzzzzzzz  # type: ignore[attr-defined] # noqa: B018


def test_resolution_is_cached_into_the_module_namespace() -> None:
    namespace: dict[str, object] = {}
    exports = build_export_map({"anycode.types": ("TokenUsage",)})
    first = lazy_getattr("anycode", "TokenUsage", exports, namespace)
    assert namespace["TokenUsage"] is first
    assert lazy_getattr("anycode", "TokenUsage", exports, namespace) is first


def test_build_export_map_applies_aliases() -> None:
    exports = build_export_map({"anycode.types": ("TokenUsage",)}, {"Usage": ("anycode.types", "TokenUsage")})
    assert exports["Usage"] == ("anycode.types", "TokenUsage")
    assert exports["TokenUsage"] == ("anycode.types", "TokenUsage")


def test_broken_mapping_reports_a_packaging_bug() -> None:
    exports = build_export_map({"anycode.types": ("NotARealSymbol",)})
    with pytest.raises(AttributeError, match="packaging bug"):
        lazy_getattr("anycode", "NotARealSymbol", exports, {})


def test_star_import_binds_every_public_name() -> None:
    namespace: dict[str, object] = {}
    exec("from anycode import *", namespace)  # noqa: S102 - exercising the real import path
    for name in anycode.__all__:
        assert name in namespace


def _probe(expression: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", expression],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_bare_import_is_cheap() -> None:
    reported = _probe("import time, sys; start = time.perf_counter(); import anycode; print(f'{time.perf_counter() - start:.4f} {len(sys.modules)}')")
    elapsed, modules = reported.split()
    assert float(elapsed) < MAX_IMPORT_SECONDS
    assert int(modules) < MAX_IMPORTED_MODULES


def test_bare_import_pulls_no_optional_dependency() -> None:
    reported = _probe(f"import sys, anycode; print(','.join(m for m in {HEAVY_DEPENDENCIES!r} if m in sys.modules))")
    assert reported == ""


@pytest.mark.parametrize("symbol", ["Agent", "Crew", "Workflow", "AnyCode", "tool"])
def test_core_symbols_pull_no_optional_dependency(symbol: str) -> None:
    reported = _probe(f"import sys, anycode; anycode.{symbol}; print(','.join(m for m in {HEAVY_DEPENDENCIES!r} if m in sys.modules))")
    assert reported == ""


def test_submodules_import_standalone() -> None:
    """Each of these closed an import cycle that the old eager package __init__ masked."""
    for module in ("anycode.types", "anycode.helpers", "anycode.contracts", "anycode.identity", "anycode.core", "anycode.memory"):
        _probe(f"import {module}")
