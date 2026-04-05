# 2.1 Pluggable Memory Backend

> **Priority:** High
> **Complexity:** Medium
> **Status:** Complete
> **Phase:** 2 — Persistence & Resilience

## Problem

The current `InMemoryStore` loses all data when the process exits. Agents cannot recall information from past sessions, teams lose shared knowledge, and there is no semantic search capability. This prevents:
- Long-running agent workflows that span multiple sessions
- Knowledge accumulation across task executions
- Contextual retrieval of relevant past interactions

## Solution

Introduce a layered memory system with three backend tiers:
1. **KV Store** — persistent key-value storage (SQLite, Redis)
2. **Vector Store** — semantic similarity search (in-memory, ChromaDB, pgvector)
3. **Composite Memory** — unified interface that queries both KV and vector stores

All backends implement the existing `MemoryStore` and new `VectorStore` Protocols.

## Files to Create

```
src/anycode/memory/
├── __init__.py
├── sqlite_store.py     # Async SQLite-backed MemoryStore
├── redis_store.py      # Redis-backed MemoryStore (optional dep)
├── vector_store.py     # VectorStore Protocol + in-memory implementation
├── chromadb_store.py   # ChromaDB-backed VectorStore (optional dep)
└── composite.py        # Unified memory that queries KV + vector
```

## Files to Modify

| File | Change |
|------|--------|
| `src/anycode/types.py` | Add `VectorStore` Protocol, `VectorSearchResult` model |
| `src/anycode/collaboration/shared_mem.py` | Accept any `MemoryStore` backend, not just `InMemoryStore` |
| `src/anycode/collaboration/team.py` | Allow configuring memory backend via `TeamConfig` |
| `src/anycode/core/orchestrator.py` | Allow configuring default memory backend via `OrchestratorConfig` |
| `pyproject.toml` | Add `persistence`, `redis`, `vector` extras groups |

## New Types

```python
class VectorSearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    text: str
    score: float
    metadata: dict[str, Any] | None = None


@runtime_checkable
class VectorStore(Protocol):
    async def add(
        self, texts: list[str], metadata: list[dict[str, Any]] | None = None
    ) -> list[str]: ...

    async def search(
        self, query: str, top_k: int = 5
    ) -> list[VectorSearchResult]: ...

    async def delete(self, ids: list[str]) -> None: ...

    async def clear(self) -> None: ...


class MemoryConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    backend: Literal["memory", "sqlite", "redis"] = "memory"
    path: str | None = None           # SQLite file path
    url: str | None = None            # Redis URL
    vector_backend: Literal["none", "memory", "chromadb"] = "none"
    vector_path: str | None = None    # ChromaDB persist path
```

## API Design

```python
from anycode import AnyCode, TeamConfig
from anycode.memory import SQLiteStore, InMemoryVectorStore

# Persistent memory via config
engine = AnyCode(config={
    "memory": {
        "backend": "sqlite",
        "path": "./agent_memory.db",
        "vector_backend": "memory",
    },
})

# Or direct injection
store = SQLiteStore("./agent_memory.db")
team = Team(TeamConfig(
    name="team",
    agents=[...],
    shared_memory=True,
    memory_store=store,
))
```

## SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS memory_entries (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    metadata TEXT,  -- JSON
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX idx_memory_created ON memory_entries(created_at);
```

## Implementation Notes

- SQLite store uses `aiosqlite` for non-blocking I/O
- Redis store uses `redis.asyncio` client
- In-memory vector store uses cosine similarity with TF-IDF (no ML dependency)
- ChromaDB store wraps the `chromadb` client with async adapter
- All stores implement the existing `MemoryStore` Protocol — drop-in replacements
- `SharedMemory` class accepts any `MemoryStore` via constructor injection
- Migration path: existing `InMemoryStore` users need zero changes

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| SQLite contention under high concurrency | Medium | Use WAL mode; document single-writer pattern |
| ChromaDB dependency is heavy | Low | Keep as optional extra; in-memory vector is zero-dep |
| Vector search quality without embeddings | Medium | TF-IDF is sufficient for agent memory; embeddings are a future enhancement |
| Redis connection failures | Medium | Implement retry with exponential backoff |

## Tests

**File:** `tests/test_memory.py`

- [ ] SQLite store persists and retrieves entries
- [ ] SQLite store survives process restart (file-based)
- [ ] Redis store CRUD operations work
- [ ] In-memory vector store returns relevant results
- [ ] Vector search scores are ordered correctly
- [ ] Composite memory queries both KV and vector
- [ ] SharedMemory works with SQLite backend
- [ ] Memory config from dict creates correct backend
- [ ] Concurrent writes to SQLite don't corrupt data
- [ ] Empty search returns empty list (not error)

## Estimated Effort

Medium — 3-4 working sessions
