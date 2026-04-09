"""Integration tests for SQLiteStore — uses a real file-backed database."""

from __future__ import annotations

import pytest

from anycode.memory.sqlite_store import SQLiteStore


@pytest.mark.integration
class TestSQLiteStoreIntegration:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, tmp_path: object) -> None:

        db_path = f"{tmp_path}/test.db"
        store = SQLiteStore(db_path)
        await store.setup()
        try:
            await store.set("a", "1")
            await store.set("b", "2")

            got = await store.get("a")
            assert got is not None
            assert got.value == "1"

            entries = await store.list()
            keys = [e.key for e in entries]
            assert sorted(keys) == ["a", "b"]

            await store.delete("a")
            assert await store.get("a") is None

            await store.clear()
            assert await store.list() == []
        finally:
            await store.teardown()

    @pytest.mark.asyncio
    async def test_persistence_across_reconnect(self, tmp_path: object) -> None:

        db_path = f"{tmp_path}/persist.db"
        store1 = SQLiteStore(db_path)
        await store1.setup()
        await store1.set("persist-key", "survive")
        await store1.teardown()

        store2 = SQLiteStore(db_path)
        await store2.setup()
        try:
            got = await store2.get("persist-key")
            assert got is not None
            assert got.value == "survive"
        finally:
            await store2.teardown()

    @pytest.mark.asyncio
    async def test_update_preserves_created_at(self, tmp_path: object) -> None:

        db_path = f"{tmp_path}/update.db"
        store = SQLiteStore(db_path)
        await store.setup()
        try:
            await store.set("k", "v1")
            first = await store.get("k")
            assert first is not None
            original_created = first.created_at

            await store.set("k", "v2")
            second = await store.get("k")
            assert second is not None
            assert second.value == "v2"
            assert second.created_at == original_created
            assert second.updated_at is not None
        finally:
            await store.teardown()
