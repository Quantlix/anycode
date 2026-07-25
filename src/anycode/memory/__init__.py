"""Pluggable memory backends for persistent and semantic storage.

Exports resolve lazily. Optional backends appear in ``__all__`` only when their
dependency is installed, decided with ``find_spec`` — importing chromadb or redis
just to find out whether to mention them costs more than a second.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

from anycode._lazy import build_export_map, lazy_getattr

# (exported name, module, distribution to probe, extras group)
_OPTIONAL_BACKENDS: tuple[tuple[str, str, str, str], ...] = (
    ("SQLiteStore", "anycode.memory.sqlite_store", "aiosqlite", "persistence"),
    ("RedisStore", "anycode.memory.redis_store", "redis", "redis"),
    ("ChromaDBVectorStore", "anycode.memory.chromadb_store", "chromadb", "vector"),
)

_EXPORTS = build_export_map(
    {
        "anycode.memory.composite": ("CompositeMemory",),
        "anycode.memory.factory": ("create_memory_store",),
        "anycode.memory.vector_store": ("InMemoryVectorStore",),
        **{module: (name,) for name, module, _dependency, _extra in _OPTIONAL_BACKENDS},
    }
)


def __getattr__(name: str) -> Any:
    for export_name, _module, dependency, extra in _OPTIONAL_BACKENDS:
        if name == export_name and find_spec(dependency) is None:
            raise AttributeError(f"{name} requires the '{extra}' extra. Install it with: uv add 'anycode-py[{extra}]'")
    return lazy_getattr(__name__, name, _EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    # Redundant aliases mark these as deliberate re-exports; the optional backends are
    # reachable as attributes but stay out of __all__ so `import *` never trips over a
    # missing extra.
    from anycode.memory.chromadb_store import ChromaDBVectorStore as ChromaDBVectorStore
    from anycode.memory.composite import CompositeMemory
    from anycode.memory.factory import create_memory_store
    from anycode.memory.redis_store import RedisStore as RedisStore
    from anycode.memory.sqlite_store import SQLiteStore as SQLiteStore
    from anycode.memory.vector_store import InMemoryVectorStore

__all__ = [
    "CompositeMemory",
    "InMemoryVectorStore",
    "create_memory_store",
]


def available_backends() -> tuple[str, ...]:
    """Optional backend names whose dependency is installed."""
    return tuple(name for name, _module, dependency, _extra in _OPTIONAL_BACKENDS if find_spec(dependency) is not None)
