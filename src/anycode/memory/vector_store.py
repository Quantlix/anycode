"""In-memory vector store using TF-IDF cosine similarity — zero external dependencies."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from anycode.helpers.uuid7 import uuid7
from anycode.types import VectorSearchResult

_WORD_RE = re.compile(r"\w+")


class InMemoryVectorStore:
    """TF-IDF + cosine similarity vector store backed by pure Python."""

    def __init__(self) -> None:
        self._docs: dict[str, _Document] = {}
        self._idf_dirty = True
        self._idf: dict[str, float] = {}

    async def add(self, texts: list[str], metadata: list[dict[str, Any]] | None = None) -> list[str]:
        ids: list[str] = []
        for i, text in enumerate(texts):
            doc_id = str(uuid7())
            meta = metadata[i] if metadata and i < len(metadata) else None
            tokens = _tokenize(text)
            tf = _compute_tf(tokens)
            self._docs[doc_id] = _Document(id=doc_id, text=text, tokens=tokens, tf=tf, metadata=meta)
            ids.append(doc_id)
        self._idf_dirty = True
        return ids

    async def search(self, query: str, top_k: int = 5) -> list[VectorSearchResult]:
        if not self._docs:
            return []
        if self._idf_dirty:
            self._rebuild_idf()
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        query_tf = _compute_tf(query_tokens)
        query_vec = {t: tf * self._idf.get(t, 0.0) for t, tf in query_tf.items()}

        scored: list[tuple[str, float]] = []
        for doc in self._docs.values():
            doc_vec = {t: tf * self._idf.get(t, 0.0) for t, tf in doc.tf.items()}
            sim = _cosine_similarity(query_vec, doc_vec)
            if sim > 0.0:
                scored.append((doc.id, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        results: list[VectorSearchResult] = []
        for doc_id, score in scored[:top_k]:
            doc = self._docs[doc_id]
            results.append(VectorSearchResult(id=doc_id, text=doc.text, score=round(score, 6), metadata=doc.metadata))
        return results

    async def delete(self, ids: list[str]) -> None:
        for doc_id in ids:
            self._docs.pop(doc_id, None)
        self._idf_dirty = True

    async def clear(self) -> None:
        self._docs.clear()
        self._idf.clear()
        self._idf_dirty = True

    def _rebuild_idf(self) -> None:
        n = len(self._docs)
        if n == 0:
            self._idf = {}
            self._idf_dirty = False
            return
        df: Counter[str] = Counter()
        for doc in self._docs.values():
            df.update(set(doc.tokens))
        self._idf = {term: math.log((1 + n) / (1 + count)) + 1.0 for term, count in df.items()}
        self._idf_dirty = False


class _Document:
    __slots__ = ("id", "text", "tokens", "tf", "metadata")

    def __init__(self, id: str, text: str, tokens: list[str], tf: dict[str, float], metadata: dict[str, Any] | None) -> None:
        self.id = id
        self.text = text
        self.tokens = tokens
        self.tf = tf
        self.metadata = metadata


def _tokenize(text: str) -> list[str]:
    return [m.group().lower() for m in _WORD_RE.finditer(text)]


def _compute_tf(tokens: list[str]) -> dict[str, float]:
    counts = Counter(tokens)
    total = len(tokens) or 1
    return {t: c / total for t, c in counts.items()}


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(a[k] * b[k] for k in a if k in b)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
