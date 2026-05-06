"""Index agent outputs and tool results into a vector store for later RAG retrieval."""

from __future__ import annotations

from datetime import UTC, datetime

from anycode.types import AgentRunResult, RAGConfig, VectorStore

INDEX_CHUNK_TARGET_CHARS = 2000


class RAGIndexer:
    """Chunks and indexes agent outputs into a vector store."""

    def __init__(self, store: VectorStore, config: RAGConfig) -> None:
        self._store = store
        self._config = config

    async def index_agent_result(self, agent_name: str, prompt: str, result: AgentRunResult) -> list[str]:
        """Index an agent's output. Returns the IDs of the inserted entries."""
        if not self._config.enabled or not self._config.auto_index:
            return []
        if not result.success or not result.output.strip():
            return []

        ids: list[str] = []
        timestamp = datetime.now(UTC).isoformat()

        chunks = _chunk(result.output, INDEX_CHUNK_TARGET_CHARS)
        metadatas = [
            {
                "source": f"agent:{agent_name}",
                "namespace": self._config.namespace,
                "timestamp": timestamp,
                "prompt": prompt[:200],
            }
            for _ in chunks
        ]
        if chunks:
            ids.extend(await self._store.add(chunks, metadatas))

        if self._config.index_tool_results and result.tool_calls:
            tool_chunks: list[str] = []
            tool_metas: list[dict[str, object]] = []
            for record in result.tool_calls:
                tool_text = f"Tool: {record.tool_name}\nOutput: {record.output[:1500]}"
                tool_chunks.append(tool_text)
                tool_metas.append(
                    {
                        "source": f"tool:{record.tool_name}",
                        "namespace": self._config.namespace,
                        "timestamp": timestamp,
                    }
                )
            if tool_chunks:
                ids.extend(await self._store.add(tool_chunks, tool_metas))

        return ids


def _chunk(text: str, target_chars: int) -> list[str]:
    """Split text on paragraph boundaries, respecting target chunk size."""
    if len(text) <= target_chars:
        return [text]
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""
    for p in paragraphs:
        if not current:
            current = p
        elif len(current) + len(p) + 2 <= target_chars:
            current = f"{current}\n\n{p}"
        else:
            chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks
