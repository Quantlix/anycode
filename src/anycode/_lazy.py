"""Lazy attribute resolution for the top-level package.

Importing ``anycode`` used to pull in every subsystem — vector stores, redis, boto3,
OpenTelemetry, provider SDKs — whether a program touched them or not. Exports are
resolved on first access instead, then cached into the module namespace so repeat
lookups are a plain dict hit.
"""

from __future__ import annotations

import difflib
import importlib
from collections.abc import Mapping
from typing import Any

MAX_SUGGESTIONS = 3
SUGGESTION_CUTOFF = 0.6


def build_export_map(
    module_exports: Mapping[str, tuple[str, ...]],
    aliases: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, tuple[str, str]]:
    """Flatten ``{module: (name, ...)}`` plus ``{alias: (module, original)}`` into ``{name: (module, attribute)}``."""
    exports: dict[str, tuple[str, str]] = {}
    for module, names in module_exports.items():
        for name in names:
            exports[name] = (module, name)
    if aliases:
        exports.update(aliases)
    return exports


def lazy_getattr(
    package: str,
    name: str,
    exports: Mapping[str, tuple[str, str]],
    namespace: dict[str, Any],
) -> Any:
    """Resolve *name* from *exports*, caching the result into *namespace*."""
    try:
        module_name, attribute = exports[name]
    except KeyError:
        raise AttributeError(_unknown_attribute_message(package, name, exports)) from None

    module = importlib.import_module(module_name)
    try:
        value = getattr(module, attribute)
    except AttributeError:
        raise AttributeError(
            f"{package}.{name} is declared as {module_name}.{attribute}, but that module has no such attribute. "
            "This is a packaging bug — please report it."
        ) from None

    namespace[name] = value
    return value


def _unknown_attribute_message(package: str, name: str, exports: Mapping[str, tuple[str, str]]) -> str:
    close = difflib.get_close_matches(name, exports, n=MAX_SUGGESTIONS, cutoff=SUGGESTION_CUTOFF)
    if close:
        return f"module '{package}' has no attribute '{name}'. Did you mean: {', '.join(close)}?"
    return f"module '{package}' has no attribute '{name}'. Run `anycode api` or see dir(anycode) for the public surface."
