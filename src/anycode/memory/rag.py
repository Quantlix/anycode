"""RAG retrieval pipeline — query vector store and inject context into agent prompts."""

from __future__ import annotations

from datetime import UTC, datetime

from anycode.types import RAGConfig, RAGContext, RAGEntry, VectorStore

CHARS_PER_TOKEN_HEURISTIC = 4


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN_HEURISTIC)


class RAGRetriever:
    """Retrieves relevant context from a vector store for prompt augmentation."""

    def __init__(self, store: VectorStore, config: RAGConfig) -> None:
        self._store = store
        self._config = config
        self._seen_ids: set[str] = set()

    async def retrieve(self, query: str, *, token_budget: int | None = None) -> RAGContext:
        if not self._config.enabled:
            return RAGContext(entries=[], total_tokens=0)

        max_context_tokens = self._config.max_context_tokens if token_budget is None else token_budget
        if max_context_tokens is not None and max_context_tokens <= 0:
            return RAGContext(entries=[], total_tokens=0)

        results = await self._store.search(query, top_k=self._config.top_k)
        entries: list[RAGEntry] = []
        total_tokens = 0

        for r in results:
            if r.score < self._config.min_relevance:
                continue
            if r.id in self._seen_ids:
                continue

            metadata = r.metadata or {}
            namespace = metadata.get("namespace")
            if namespace and namespace != self._config.namespace:
                continue

            text_tokens = _estimate_tokens(r.text)
            if max_context_tokens is not None and total_tokens + text_tokens > max_context_tokens:
                break

            self._seen_ids.add(r.id)
            entries.append(
                RAGEntry(
                    text=r.text,
                    source=str(metadata.get("source", "memory")),
                    score=r.score,
                    timestamp=_parse_ts(metadata.get("timestamp")),
                )
            )
            total_tokens += text_tokens

        return RAGContext(entries=entries, total_tokens=total_tokens)

    @staticmethod
    def format_context(context: RAGContext) -> str:
        """Format retrieved context as a system-prompt-friendly block."""
        if not context.entries:
            return ""
        lines: list[str] = ["## Relevant Context from Past Sessions"]
        for e in context.entries:
            ts = e.timestamp.date().isoformat() if isinstance(e.timestamp, datetime) else "unknown"
            lines.append(f"\n[{e.source}, {ts}] (relevance={e.score:.2f})")
            lines.append(e.text)
        return "\n".join(lines)


def _parse_ts(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.now(UTC)
