"""ChromaDB-backed VectorStore with async adapter."""

from __future__ import annotations

import asyncio
from typing import Any

from anycode.constants import CHROMADB_DEFAULT_COLLECTION, CHROMADB_DEFAULT_PORT
from anycode.helpers.uuid7 import uuid7

try:
    import chromadb
except ImportError as exc:
    raise ImportError("ChromaDBVectorStore requires the 'chromadb' package. Install it with: pip install anycode-py[vector]") from exc

from anycode.types import VectorSearchResult


class ChromaDBVectorStore:
    """ChromaDB VectorStore implementation — runs blocking calls in asyncio.to_thread()."""

    def __init__(self, path: str | None = None, collection_name: str = CHROMADB_DEFAULT_COLLECTION, url: str | None = None) -> None:
        self._path = path
        self._url = url
        self._collection_name = collection_name
        self._client: Any = None
        self._collection: Any = None

    async def setup(self) -> None:
        def _init() -> tuple[Any, Any]:
            if self._url:
                host = self._url.split("://")[-1].split(":")[0]
                raw_port = self._url.rsplit(":", 1)[-1].split("/")[0]
                port = int(raw_port) if ":" in self._url.rsplit("//", 1)[-1] else CHROMADB_DEFAULT_PORT
                client = chromadb.HttpClient(host=host, port=port)
            elif self._path:
                client = chromadb.PersistentClient(path=self._path)
            else:
                client = chromadb.EphemeralClient()
            collection = client.get_or_create_collection(name=self._collection_name)
            return client, collection

        self._client, self._collection = await asyncio.to_thread(_init)

    async def teardown(self) -> None:
        self._collection = None
        self._client = None

    def _coll(self) -> Any:
        if self._collection is None:
            raise RuntimeError("ChromaDBVectorStore not initialized. Call setup() first.")
        return self._collection

    async def add(self, texts: list[str], metadata: list[dict[str, Any]] | None = None) -> list[str]:
        ids = [str(uuid7()) for _ in texts]
        coll = self._coll()
        kwargs: dict[str, Any] = {"ids": ids, "documents": texts}
        if metadata:
            kwargs["metadatas"] = metadata
        await asyncio.to_thread(coll.add, **kwargs)
        return ids

    async def search(self, query: str, top_k: int = 5) -> list[VectorSearchResult]:
        coll = self._coll()
        raw = await asyncio.to_thread(coll.query, query_texts=[query], n_results=top_k)
        results: list[VectorSearchResult] = []
        if not raw or not raw.get("ids"):
            return results
        doc_ids = raw["ids"][0]
        documents = raw.get("documents", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        for i, doc_id in enumerate(doc_ids):
            score = 1.0 / (1.0 + distances[i]) if distances else 0.0
            results.append(
                VectorSearchResult(
                    id=doc_id,
                    text=documents[i] if documents else "",
                    score=round(score, 6),
                    metadata=dict(metadatas[i]) if metadatas else None,
                )
            )
        return results

    async def delete(self, ids: list[str]) -> None:
        coll = self._coll()
        await asyncio.to_thread(coll.delete, ids=ids)

    async def clear(self) -> None:
        if self._client is None:
            return
        client = self._client
        name = self._collection_name
        await asyncio.to_thread(client.delete_collection, name)
        self._collection = await asyncio.to_thread(client.get_or_create_collection, name=name)
