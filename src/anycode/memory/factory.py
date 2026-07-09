"""Factory for creating memory store instances from MemoryConfig."""

from __future__ import annotations

import logging

from anycode.collaboration.kv_store import InMemoryStore
from anycode.memory.vector_store import InMemoryVectorStore
from anycode.types import MemoryConfig, MemoryStore, VectorStore

logger = logging.getLogger(__name__)


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


def create_vector_store(config: MemoryConfig | None = None) -> VectorStore:
    """Create the vector store backend selected by `MemoryConfig.vector_backend`.

    With no config (or backend "none"/"memory") this falls back to the
    in-process TF-IDF store — retrieval works, but long-term memory does not
    survive a restart, which is why the fallback is logged loudly.
    """
    backend = config.vector_backend if config else "none"

    if backend == "chromadb":
        try:
            from anycode.memory.chromadb_store import ChromaDBVectorStore
        except ImportError as exc:
            raise ImportError("ChromaDB backend requires: pip install anycode-py[vector]") from exc
        return ChromaDBVectorStore(path=config.vector_path if config else None)

    if backend == "none":
        logger.warning(
            "RAG memory is using the in-memory vector store: long-term memory will "
            "not survive a process restart. Set MemoryConfig(vector_backend='chromadb') "
            "for persistent memory."
        )
    return InMemoryVectorStore()
