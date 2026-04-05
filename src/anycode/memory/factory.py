"""Factory for creating memory store instances from MemoryConfig."""

from __future__ import annotations

from anycode.collaboration.kv_store import InMemoryStore
from anycode.types import MemoryConfig, MemoryStore


def create_memory_store(config: MemoryConfig | dict[str, object] | None = None) -> MemoryStore:
    """Create the appropriate MemoryStore backend from a MemoryConfig or dict."""
    if config is None:
        return InMemoryStore()

    cfg = MemoryConfig.model_validate(config) if isinstance(config, dict) else config

    if cfg.backend == "memory":
        return InMemoryStore()

    if cfg.backend == "sqlite":
        try:
            from anycode.memory.sqlite_store import SQLiteStore
        except ImportError as exc:
            raise ImportError("SQLite backend requires: pip install anycode-py[persistence]") from exc
        return SQLiteStore(path=cfg.path or ":memory:")

    if cfg.backend == "redis":
        try:
            from anycode.memory.redis_store import RedisStore
        except ImportError as exc:
            raise ImportError("Redis backend requires: pip install anycode-py[redis]") from exc
        return RedisStore(url=cfg.url or "redis://localhost:6379/0")

    raise ValueError(f"Unknown memory backend: {cfg.backend!r}")
