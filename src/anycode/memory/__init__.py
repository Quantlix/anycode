"""Pluggable memory backends for persistent and semantic storage."""

from anycode.memory.composite import CompositeMemory
from anycode.memory.factory import create_memory_store
from anycode.memory.vector_store import InMemoryVectorStore

__all__ = [
    "CompositeMemory",
    "InMemoryVectorStore",
    "create_memory_store",
]

# Optional backends — re-exported when their dependencies are available.
try:
    from anycode.memory.sqlite_store import SQLiteStore  # noqa: F401

    __all__.append("SQLiteStore")
except ImportError:
    pass

try:
    from anycode.memory.redis_store import RedisStore  # noqa: F401

    __all__.append("RedisStore")
except ImportError:
    pass

try:
    from anycode.memory.chromadb_store import ChromaDBVectorStore  # noqa: F401

    __all__.append("ChromaDBVectorStore")
except ImportError:
    pass
