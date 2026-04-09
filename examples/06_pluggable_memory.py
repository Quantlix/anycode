# Demo 06 — Pluggable Memory: SQLite, Redis, Vector Search, ChromaDB, Composite
# Execute: uv run python examples/06_pluggable_memory.py
#
# Demonstrates:
#   1. SQLite persistent KV store — CRUD, persistence across reconnects
#   2. Redis KV store — CRUD operations against live Redis
#   3. In-memory vector store — TF-IDF semantic search (zero deps)
#   4. ChromaDB vector store — embedding-backed semantic search
#   5. Composite memory — unified KV + vector search
#   6. SharedMemory DI — inject any MemoryStore backend
#   7. Memory factory — create stores from config dicts
#
# Requires: Docker services running (docker compose up -d)
#   Redis  → localhost:6380
#   ChromaDB → localhost:8100

import asyncio
import os
import socket
import sys
import tempfile
import uuid
from io import StringIO
from pathlib import Path

from anycode.collaboration.shared_mem import SharedMemory
from anycode.memory.chromadb_store import ChromaDBVectorStore
from anycode.memory.composite import CompositeMemory
from anycode.memory.factory import create_memory_store
from anycode.memory.redis_store import RedisStore
from anycode.memory.sqlite_store import SQLiteStore
from anycode.memory.vector_store import InMemoryVectorStore
from anycode.types import MemoryConfig

SEPARATOR = "=" * 60
REDIS_URL = os.environ.get("ANYCODE_TEST_REDIS_URL", "redis://localhost:6380/0")
CHROMADB_URL = os.environ.get("ANYCODE_TEST_CHROMADB_URL", "http://localhost:8100")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def _is_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# --- Section 1: SQLite Persistent KV Store ---


async def demo_sqlite_store() -> None:
    print(f"\n{SEPARATOR}")
    print("  1. SQLite Persistent KV Store")
    print(SEPARATOR)

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "memory.db")

        # --- 1a. Basic CRUD ---
        print("\n--- CRUD Operations ---")
        store = SQLiteStore(db_path)
        await store.setup()

        await store.set("agent:preferences", "Prefer concise responses", {"source": "config"})
        await store.set("session:context", "Working on Python project", {"topic": "dev"})
        await store.set("fact:python", "Python 3.12 supports better type hints")

        entry = await store.get("agent:preferences")
        assert entry is not None
        print(f"GET agent:preferences → key={entry.key}, value={entry.value!r}")
        print(f"  metadata={entry.metadata}, created_at={entry.created_at}")

        entries = await store.list()
        print(f"LIST → {len(entries)} entries: {[e.key for e in entries]}")

        await store.delete("session:context")
        entries = await store.list()
        print(f"DELETE session:context → {len(entries)} entries remain")

        # --- 1b. Update preserves created_at ---
        print("\n--- Update Preserves created_at ---")
        original = await store.get("fact:python")
        assert original is not None
        await store.set("fact:python", "Python 3.13 improves error messages")
        updated = await store.get("fact:python")
        assert updated is not None
        print(f"Original created_at: {original.created_at}")
        print(f"Updated created_at:  {updated.created_at}  (same={original.created_at == updated.created_at})")
        print(f"Updated value: {updated.value!r}")

        await store.teardown()

        # --- 1c. Persistence across reconnect ---
        print("\n--- Persistence Across Reconnect ---")
        store2 = SQLiteStore(db_path)
        await store2.setup()
        surviving = await store2.get("agent:preferences")
        assert surviving is not None
        print(f"After reconnect: {surviving.key}={surviving.value!r}  ✓ survived")

        surviving_fact = await store2.get("fact:python")
        assert surviving_fact is not None
        print(f"After reconnect: {surviving_fact.key}={surviving_fact.value!r}  ✓ survived")

        # --- 1d. Concurrent writes ---
        print("\n--- Concurrent Writes (10 parallel) ---")
        await store2.clear()

        async def _write(i: int) -> None:
            await store2.set(f"concurrent-{i}", f"value-{i}")

        await asyncio.gather(*[_write(i) for i in range(10)])
        all_entries = await store2.list()
        print(f"Concurrent writes: {len(all_entries)} entries (expected 10)  ✓")

        await store2.teardown()


# --- Section 2: Redis KV Store ---


async def demo_redis_store() -> None:
    print(f"\n{SEPARATOR}")
    print("  2. Redis KV Store")
    print(SEPARATOR)

    if not _is_port_open("localhost", 6380):
        print("  SKIPPED — Redis not available at localhost:6380")
        return

    store = RedisStore(url=REDIS_URL)
    await store.setup()

    try:
        # Clean slate
        await store.clear()

        # CRUD
        print("\n--- CRUD Operations ---")
        await store.set("redis:agent", "Agent state data", {"ttl": "3600"})
        await store.set("redis:cache", "Cached LLM response")
        await store.set("redis:fact", "Redis supports pub/sub")

        entry = await store.get("redis:agent")
        assert entry is not None
        print(f"GET redis:agent → {entry.value!r}, metadata={entry.metadata}")

        entries = await store.list()
        print(f"LIST → {len(entries)} entries: {[e.key for e in entries]}")

        await store.delete("redis:cache")
        remaining = await store.list()
        print(f"DELETE redis:cache → {len(remaining)} entries remain")

        # Update preserves created_at
        print("\n--- Update Preserves created_at ---")
        original = await store.get("redis:fact")
        assert original is not None
        await store.set("redis:fact", "Redis supports streams and pub/sub")
        updated = await store.get("redis:fact")
        assert updated is not None
        print(f"Original created_at: {original.created_at}")
        print(f"Updated created_at:  {updated.created_at}  (same={original.created_at == updated.created_at})")

        # Get non-existent
        missing = await store.get("does-not-exist")
        print(f"\nGET non-existent → {missing}  ✓ (returns None)")

    finally:
        await store.clear()
        await store.teardown()


# --- Section 3: In-Memory Vector Store (TF-IDF) ---


async def demo_vector_store() -> None:
    print(f"\n{SEPARATOR}")
    print("  3. In-Memory Vector Store (TF-IDF)")
    print(SEPARATOR)

    vs = InMemoryVectorStore()

    # Add documents with metadata
    print("\n--- Adding Documents ---")
    docs = [
        "Python is a versatile programming language used for web development",
        "Machine learning models require large datasets for training",
        "Docker containers provide lightweight application isolation",
        "Redis is an in-memory data store used for caching",
        "PostgreSQL is a powerful open-source relational database",
        "Kubernetes orchestrates container deployments at scale",
        "FastAPI is a modern Python web framework with async support",
        "Neural networks learn patterns through backpropagation",
    ]
    metadata = [{"topic": t} for t in ["python", "ml", "docker", "redis", "postgres", "k8s", "python", "ml"]]
    ids = await vs.add(docs, metadata)
    print(f"Added {len(ids)} documents")

    # Search
    print("\n--- Semantic Search ---")
    queries = [
        "Python web framework",
        "container orchestration",
        "database storage",
        "deep learning training",
    ]
    for query in queries:
        results = await vs.search(query, top_k=3)
        print(f"\nQuery: {query!r}")
        for i, r in enumerate(results):
            print(f"  [{i + 1}] score={r.score:.4f} | {r.text[:70]}... | meta={r.metadata}")

    # Edge cases
    print("\n--- Edge Cases ---")
    empty_results = await vs.search("completely unrelated quantum physics", top_k=3)
    print(f"Unrelated query → {len(empty_results)} results (scores: {[r.score for r in empty_results]})")

    # Delete and verify
    print("\n--- Delete ---")
    await vs.delete([ids[0]])
    after_delete = await vs.search("Python programming", top_k=3)
    original_ids = {r.id for r in after_delete}
    print(f"Deleted first doc, search hits: {len(after_delete)} (original ID absent: {ids[0] not in original_ids})")

    # Clear
    await vs.clear()
    after_clear = await vs.search("anything", top_k=5)
    print(f"After clear: {len(after_clear)} results  ✓")


# --- Section 4: ChromaDB Vector Store ---


async def demo_chromadb_store() -> None:
    print(f"\n{SEPARATOR}")
    print("  4. ChromaDB Vector Store")
    print(SEPARATOR)

    if not _is_port_open("localhost", 8100):
        print("  SKIPPED — ChromaDB not available at localhost:8100")
        return

    collection = f"demo_{uuid.uuid4().hex[:8]}"
    store = ChromaDBVectorStore(url=CHROMADB_URL, collection_name=collection)
    await store.setup()

    try:
        # Add documents
        print("\n--- Adding Documents ---")
        docs = [
            "Async Python uses coroutines and event loops for concurrency",
            "React is a JavaScript library for building user interfaces",
            "PostgreSQL supports JSON columns and full-text search",
            "Kubernetes pods are the smallest deployable units",
            "TensorFlow is a machine learning framework by Google",
        ]
        meta = [{"lang": "python"}, {"lang": "js"}, {"lang": "sql"}, {"lang": "yaml"}, {"lang": "python"}]
        ids = await store.add(docs, meta)
        print(f"Added {len(ids)} documents to ChromaDB collection: {collection}")

        # Search
        print("\n--- Semantic Search (ChromaDB embeddings) ---")
        queries = ["Python async programming", "frontend framework", "database features"]
        for query in queries:
            results = await store.search(query, top_k=2)
            print(f"\nQuery: {query!r}")
            for i, r in enumerate(results):
                print(f"  [{i + 1}] score={r.score:.4f} | {r.text[:60]}... | meta={r.metadata}")

        # Delete
        print("\n--- Delete ---")
        await store.delete([ids[0]])
        after = await store.search("async coroutines", top_k=5)
        deleted_ids = {r.id for r in after}
        print(f"Deleted first doc. ID still in results: {ids[0] in deleted_ids}")

    finally:
        await store.clear()
        await store.teardown()


# --- Section 5: Composite Memory ---


async def demo_composite_memory() -> None:
    print(f"\n{SEPARATOR}")
    print("  5. Composite Memory (KV + Vector)")
    print(SEPARATOR)

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "composite.db")

        kv = SQLiteStore(db_path)
        await kv.setup()
        vs = InMemoryVectorStore()
        composite = CompositeMemory(kv_store=kv, vector_store=vs, auto_index=True)

        try:
            # Set entries (auto-indexed into vector store)
            print("\n--- Set + Auto-Index ---")
            await composite.set("meeting:standup", "Discussed deployment pipeline and CI improvements")
            await composite.set("meeting:retro", "Team wants better code review process")
            await composite.set("decision:arch", "Chose event-driven architecture for notifications")
            await composite.set("note:perf", "Redis caching reduced P99 latency by 40%")
            print("Set 4 entries (auto-indexed into vector store)")

            # KV retrieval
            print("\n--- KV Get ---")
            entry = await composite.get("decision:arch")
            assert entry is not None
            print(f"GET decision:arch → {entry.value!r}")

            # Semantic search
            print("\n--- Semantic Search via Composite ---")
            results = await composite.search("deployment and CI", top_k=2)
            print(f"Query: 'deployment and CI' → {len(results)} result(s)")
            for r in results:
                print(f"  score={r.score:.4f} | {r.text[:60]}...")

            results2 = await composite.search("performance optimization", top_k=2)
            print(f"\nQuery: 'performance optimization' → {len(results2)} result(s)")
            for r in results2:
                print(f"  score={r.score:.4f} | {r.text[:60]}...")

            # List all
            all_entries = await composite.list()
            print(f"\nLIST → {len(all_entries)} entries")

            # Clear
            await composite.clear()
            after_clear = await composite.get("meeting:standup")
            vs_after = await composite.search("anything", top_k=5)
            print(f"After clear: KV={after_clear}, vector hits={len(vs_after)}  ✓")

        finally:
            await kv.teardown()


# --- Section 6: SharedMemory with Backend DI ---


async def demo_shared_memory_di() -> None:
    print(f"\n{SEPARATOR}")
    print("  6. SharedMemory with Backend Injection")
    print(SEPARATOR)

    # Default (in-memory)
    print("\n--- Default InMemoryStore ---")
    mem_default = SharedMemory()
    await mem_default.write("agent-1", "context", "Working on task A")
    entry = await mem_default.read("agent-1/context")
    assert entry is not None
    print(f"Default store: {entry.value!r}")
    print(f"Store type: {type(mem_default.get_store()).__name__}")

    # Injected SQLite store
    print("\n--- Injected SQLiteStore ---")
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "shared.db")
        sqlite_store = SQLiteStore(db_path)
        await sqlite_store.setup()

        try:
            mem_sqlite = SharedMemory(store=sqlite_store)
            await mem_sqlite.write("agent-1", "plan", "Parse CSV, validate, insert into DB")
            await mem_sqlite.write("agent-2", "status", "Waiting for agent-1 output")

            entry1 = await mem_sqlite.read("agent-1/plan")
            entry2 = await mem_sqlite.read("agent-2/status")
            assert entry1 is not None
            assert entry2 is not None
            print(f"agent-1/plan → {entry1.value!r}")
            print(f"agent-2/status → {entry2.value!r}")
            print(f"Store type: {type(mem_sqlite.get_store()).__name__}")

            # Verify data is in the raw SQLite store too
            raw = await sqlite_store.get("agent-1/plan")
            assert raw is not None
            print(f"Direct SQLite get confirms: {raw.value!r}  ✓")

        finally:
            await sqlite_store.teardown()


# --- Section 7: Memory Factory ---


async def demo_memory_factory() -> None:
    print(f"\n{SEPARATOR}")
    print("  7. Memory Factory (Config-Driven)")
    print(SEPARATOR)

    # Default
    print("\n--- Default (in-memory) ---")
    store1 = create_memory_store()
    print(f"create_memory_store() → {type(store1).__name__}")

    # From dict
    print("\n--- From dict ---")
    store2 = create_memory_store({"backend": "memory"})
    print(f"From dict {{'backend': 'memory'}} → {type(store2).__name__}")

    # SQLite from config
    print("\n--- SQLite from MemoryConfig ---")
    store3 = create_memory_store(MemoryConfig(backend="sqlite", path=":memory:"))
    print(f"MemoryConfig(backend='sqlite') → {type(store3).__name__}")

    # Redis from config (if available)
    if _is_port_open("localhost", 6380):
        print("\n--- Redis from MemoryConfig ---")
        store4 = create_memory_store(MemoryConfig(backend="redis", url=REDIS_URL))
        print(f"MemoryConfig(backend='redis') → {type(store4).__name__}")
    else:
        print("\n--- Redis from MemoryConfig --- SKIPPED (no Redis)")


# --- Main ---


class _OutputCapture:
    def __init__(self) -> None:
        self._buffer = StringIO()
        self._stdout = sys.stdout

    def write(self, text: str) -> None:
        self._stdout.write(text)
        self._buffer.write(text)

    def flush(self) -> None:
        self._stdout.flush()

    def get_output(self) -> str:
        return self._buffer.getvalue()


async def main() -> None:
    capture = _OutputCapture()
    sys.stdout = capture  # type: ignore[assignment]

    print("AnyCode — Pluggable Memory Demo ")
    print(f"Redis: {'✓' if _is_port_open('localhost', 6380) else '✗'} | ChromaDB: {'✓' if _is_port_open('localhost', 8100) else '✗'}")

    await demo_sqlite_store()
    await demo_redis_store()
    await demo_vector_store()
    await demo_chromadb_store()
    await demo_composite_memory()
    await demo_shared_memory_di()
    await demo_memory_factory()

    print(f"\n{SEPARATOR}")
    print("  All memory demos completed successfully!")
    print(SEPARATOR)

    sys.stdout = capture._stdout
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "06-pluggable-memory-output.txt"
    output_file.write_text(capture.get_output(), encoding="utf-8")
    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
