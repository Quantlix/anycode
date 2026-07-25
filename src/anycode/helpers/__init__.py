"""Small shared utilities.

Exports resolve lazily: ``usage_tracker`` depends on ``anycode.types``, which depends on
``anycode.identity`` and ``anycode.contracts``, which depend back on this package.
Re-exporting eagerly closes that loop and makes ``import anycode.types`` fail on its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anycode._lazy import build_export_map, lazy_getattr

_EXPORTS = build_export_map(
    {
        "anycode.helpers.concurrency_gate": ("Semaphore",),
        "anycode.helpers.usage_tracker": ("EMPTY_USAGE", "merge_usage"),
        "anycode.helpers.uuid7": ("uuid7",),
    }
)


def __getattr__(name: str) -> Any:
    return lazy_getattr(__name__, name, _EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from anycode.helpers.concurrency_gate import Semaphore
    from anycode.helpers.usage_tracker import EMPTY_USAGE, merge_usage
    from anycode.helpers.uuid7 import uuid7

__all__ = ["EMPTY_USAGE", "Semaphore", "merge_usage", "uuid7"]
