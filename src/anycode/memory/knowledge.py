"""Knowledge store: curated "what was learned", separate from conversation logs.

Distilled insight must survive compaction and transcript roll-off, so it lives
in its own tier: plain Markdown files with YAML-style frontmatter in a
user-visible directory — human-readable, human-editable, greppable, diffable.

Governance rules for long-lived writable state:

* Entries are written by explicit save operations (deliberate sedimentation),
  never by automatic absorption.
* Every entry carries provenance (source, author, timestamp, content hash).
* Entries are append-only: curation supersedes an entry with a new one rather
  than mutating it in place, preserving the audit trail.

Any derived index (e.g. a vector index) is rebuildable from these files; the
files are ground truth.
"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from anycode.helpers.uuid7 import uuid7
from anycode.security.redaction import redact_text

if TYPE_CHECKING:
    from anycode.types import ToolDefinition

KnowledgeSource = str  # "user" | "agent" | "tool" | "retrieved"


class KnowledgeEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    title: str
    content: str
    tags: tuple[str, ...] = ()
    source: KnowledgeSource = "agent"
    author: str = ""
    created_at: datetime
    content_hash: str
    supersedes: str | None = None
    superseded_by: str | None = None


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


class KnowledgeStore:
    """Markdown + frontmatter knowledge files under a single directory."""

    def __init__(self, root: str | Path = ".anycode/knowledge", *, redact_sensitive_data: bool = True) -> None:
        self._root = Path(root)
        self._redact_sensitive_data = redact_sensitive_data

    @property
    def root(self) -> Path:
        return self._root

    def save(
        self,
        title: str,
        content: str,
        *,
        tags: tuple[str, ...] = (),
        source: KnowledgeSource = "agent",
        author: str = "",
        supersedes: str | None = None,
    ) -> KnowledgeEntry:
        """Persist one learned fact/decision/lesson as its own file.

        When `supersedes` names an existing entry, that entry is kept but
        marked superseded — append-plus-supersede, never in-place mutation.
        """
        self._root.mkdir(parents=True, exist_ok=True)
        if self._redact_sensitive_data:
            title = redact_text(title)
            content = redact_text(content)
            tags = tuple(redact_text(tag) for tag in tags)
            author = redact_text(author)
        entry = KnowledgeEntry(
            id=str(uuid7()),
            title=title,
            content=content,
            tags=tags,
            source=source,
            author=author,
            created_at=datetime.now(UTC),
            content_hash=_hash_content(content),
            supersedes=supersedes,
        )
        self._write_entry(entry)
        if supersedes:
            old = self.get(supersedes)
            if old is not None:
                self._write_entry(old.model_copy(update={"superseded_by": entry.id}))
        return entry

    def get(self, entry_id: str) -> KnowledgeEntry | None:
        path = self._root / f"{entry_id}.md"
        if not path.exists():
            return None
        return _parse_entry(path.read_text(encoding="utf-8"))

    def list_entries(self, *, include_superseded: bool = False) -> list[KnowledgeEntry]:
        if not self._root.exists():
            return []
        entries = []
        for path in sorted(self._root.glob("*.md")):
            entry = _parse_entry(path.read_text(encoding="utf-8"))
            if entry is None:
                continue
            if entry.superseded_by and not include_superseded:
                continue
            entries.append(entry)
        return entries

    def search(self, query: str, *, top_k: int = 5) -> list[KnowledgeEntry]:
        """Keyword search over current entries — the always-available fallback
        in the retrieval degradation chain (vector -> keyword -> scan)."""
        terms = [t for t in re.split(r"\W+", query.lower()) if t]
        if not terms:
            return []
        scored: list[tuple[int, KnowledgeEntry]] = []
        for entry in self.list_entries():
            haystack = f"{entry.title}\n{entry.content}\n{' '.join(entry.tags)}".lower()
            score = sum(haystack.count(term) for term in terms)
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _score, entry in scored[:top_k]]

    def verify(self, entry_id: str) -> bool:
        """Check an entry's content against its stored hash (tamper/corruption)."""
        entry = self.get(entry_id)
        return entry is not None and _hash_content(entry.content) == entry.content_hash

    def _write_entry(self, entry: KnowledgeEntry) -> None:
        path = self._root / f"{entry.id}.md"
        tmp = path.with_suffix(".md.tmp")
        tmp.write_text(_render_entry(entry), encoding="utf-8")
        os.replace(tmp, path)


def _render_entry(entry: KnowledgeEntry) -> str:
    lines = [
        "---",
        f"id: {entry.id}",
        f"title: {entry.title}",
        f"tags: {', '.join(entry.tags)}",
        f"source: {entry.source}",
        f"author: {entry.author}",
        f"created_at: {entry.created_at.isoformat()}",
        f"content_hash: {entry.content_hash}",
        f"supersedes: {entry.supersedes or ''}",
        f"superseded_by: {entry.superseded_by or ''}",
        "---",
        "",
        entry.content,
    ]
    return "\n".join(lines) + "\n"


def _parse_entry(raw: str) -> KnowledgeEntry | None:
    match = re.match(r"^---\n(.*?)\n---\n\n?(.*)$", raw, flags=re.DOTALL)
    if not match:
        return None
    frontmatter: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        frontmatter[key.strip()] = value.strip()
    try:
        return KnowledgeEntry(
            id=frontmatter["id"],
            title=frontmatter.get("title", ""),
            content=match.group(2).rstrip("\n"),
            tags=tuple(t.strip() for t in frontmatter.get("tags", "").split(",") if t.strip()),
            source=frontmatter.get("source", "agent"),
            author=frontmatter.get("author", ""),
            created_at=frontmatter["created_at"],  # type: ignore[arg-type]
            content_hash=frontmatter.get("content_hash", ""),
            supersedes=frontmatter.get("supersedes") or None,
            superseded_by=frontmatter.get("superseded_by") or None,
        )
    except (KeyError, ValueError):
        return None


def build_knowledge_tools(store: KnowledgeStore) -> list[ToolDefinition]:
    """Opt-in agent tools for deliberate knowledge sedimentation and recall.

    `knowledge_save` writes are high-leverage persistent state — register these
    only for agents that should accumulate long-term knowledge, and gate them
    behind approval policies where the deployment demands it.
    """
    import json

    from anycode.types import ToolDefinition, ToolResult

    class KnowledgeSaveInput(BaseModel):
        title: str
        content: str
        tags: str = ""

    class KnowledgeSearchInput(BaseModel):
        query: str
        top_k: int = 5

    async def _save(*, title: str, content: str, tags: str = "", **_kwargs: object) -> ToolResult:
        entry = store.save(
            title,
            content,
            tags=tuple(t.strip() for t in tags.split(",") if t.strip()),
            source="agent",
        )
        return ToolResult(data=f"Saved knowledge entry {entry.id}: {entry.title}")

    async def _search(*, query: str, top_k: int = 5, **_kwargs: object) -> ToolResult:
        matches = store.search(query, top_k=top_k)
        payload = [{"id": e.id, "title": e.title, "tags": list(e.tags), "content": e.content[:1000]} for e in matches]
        return ToolResult(data=json.dumps(payload, default=str))

    return [
        ToolDefinition(
            name="knowledge_save",
            description=(
                "Save a durable lesson, decision, or fact to long-term knowledge. "
                "Use for insight worth keeping beyond this session — not for raw logs."
            ),
            input_model=KnowledgeSaveInput,
            execute=_save,
            side_effecting=True,
        ),
        ToolDefinition(
            name="memory_search",
            description="Search long-term knowledge for lessons, decisions, and facts from earlier work.",
            input_model=KnowledgeSearchInput,
            execute=_search,
        ),
    ]


def apply_retention(directory: str | Path, *, max_age_days: float, pattern: str = "*") -> list[Path]:
    """Delete files older than `max_age_days` (FIFO retention for rolling logs).

    Intended for session notes and archives — never for the knowledge tier,
    which is permanent and curated by supersession instead.
    """
    root = Path(directory)
    if not root.exists():
        return []
    cutoff = datetime.now(UTC).timestamp() - max_age_days * 86_400
    removed: list[Path] = []
    for path in sorted(root.glob(pattern)):
        if path.is_file() and path.stat().st_mtime < cutoff:
            path.unlink()
            removed.append(path)
    return removed
