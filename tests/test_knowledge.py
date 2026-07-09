"""Tests for tiered persistent memory: vector-backend wiring, knowledge store, retention."""

from __future__ import annotations

import os
import time
from pathlib import Path

from anycode.memory.factory import create_vector_store
from anycode.memory.knowledge import KnowledgeStore, apply_retention, build_knowledge_tools
from anycode.memory.vector_store import InMemoryVectorStore
from anycode.types import MemoryConfig

# -- vector backend wiring --


def test_create_vector_store_defaults_to_in_memory_with_warning(caplog) -> None:  # type: ignore[no-untyped-def]
    with caplog.at_level("WARNING"):
        store = create_vector_store(None)
    assert isinstance(store, InMemoryVectorStore)
    assert "will not survive a process restart" in caplog.text


def test_create_vector_store_memory_backend_no_warning(caplog) -> None:  # type: ignore[no-untyped-def]
    with caplog.at_level("WARNING"):
        store = create_vector_store(MemoryConfig(vector_backend="memory"))
    assert isinstance(store, InMemoryVectorStore)
    assert "will not survive" not in caplog.text


def test_orchestrator_honors_vector_backend() -> None:
    """The orchestrator builds its RAG store through the factory, not a hardcoded class."""
    import inspect

    from anycode.core import orchestrator

    source = inspect.getsource(orchestrator)
    assert "create_vector_store(self._config.memory)" in source
    assert "InMemoryVectorStore()" not in source


# -- knowledge store --


def test_knowledge_save_and_get_roundtrip(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path)
    entry = store.save(
        "Use WAL mode for SQLite",
        "Concurrent readers need WAL; default journal mode caused lock errors in run 42.",
        tags=("sqlite", "lesson"),
        source="agent",
        author="developer",
    )
    loaded = store.get(entry.id)
    assert loaded is not None
    assert loaded.title == "Use WAL mode for SQLite"
    assert "lock errors" in loaded.content
    assert loaded.tags == ("sqlite", "lesson")
    assert loaded.source == "agent"
    assert loaded.content_hash == entry.content_hash
    assert store.verify(entry.id)

    # Files are plain markdown with frontmatter — human-readable ground truth.
    raw = (tmp_path / f"{entry.id}.md").read_text(encoding="utf-8")
    assert raw.startswith("---\n")
    assert "Use WAL mode for SQLite" in raw


def test_knowledge_supersede_keeps_audit_trail(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path)
    old = store.save("Lesson", "v1: partially wrong")
    new = store.save("Lesson", "v2: corrected", supersedes=old.id)

    current = store.list_entries()
    assert [e.id for e in current] == [new.id]  # superseded entry hidden by default

    full = store.list_entries(include_superseded=True)
    assert len(full) == 2
    old_reloaded = store.get(old.id)
    assert old_reloaded is not None
    assert old_reloaded.superseded_by == new.id  # audit trail, not deletion
    assert old_reloaded.content == "v1: partially wrong"


def test_knowledge_search_keyword_fallback(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path)
    store.save("Deploy steps", "Use blue-green deploys on Fridays only.")
    store.save("DB tuning", "Increase pool size for batch jobs.")

    hits = store.search("blue-green deploy")
    assert len(hits) == 1
    assert hits[0].title == "Deploy steps"
    assert store.search("") == []


def test_knowledge_survives_new_store_instance(tmp_path: Path) -> None:
    """Entries persist across 'process restarts' (fresh store over same dir)."""
    KnowledgeStore(tmp_path).save("Persistent", "still here")
    fresh = KnowledgeStore(tmp_path)
    assert [e.title for e in fresh.list_entries()] == ["Persistent"]


async def test_knowledge_tools_roundtrip(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path)
    save_tool, search_tool = build_knowledge_tools(store)
    assert save_tool.name == "knowledge_save"
    assert search_tool.name == "memory_search"

    result = await save_tool.execute(title="Rate limits", content="Provider X throttles at 60 rpm.", tags="providers")
    assert "Saved knowledge entry" in result.data

    found = await search_tool.execute(query="throttles rpm")
    assert "Rate limits" in found.data


# -- retention --


def test_apply_retention_removes_only_old_files(tmp_path: Path) -> None:
    old_file = tmp_path / "2026-01-01.md"
    new_file = tmp_path / "2026-07-08.md"
    old_file.write_text("old", encoding="utf-8")
    new_file.write_text("new", encoding="utf-8")
    stale_mtime = time.time() - 60 * 86_400
    os.utime(old_file, (stale_mtime, stale_mtime))

    removed = apply_retention(tmp_path, max_age_days=50, pattern="*.md")
    assert removed == [old_file]
    assert not old_file.exists()
    assert new_file.exists()
