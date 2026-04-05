"""Integration tests for checkpoint stores with real SQLite files."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from anycode.checkpoint.store import FilesystemCheckpointStore
from anycode.tasks.task import create_task
from anycode.types import CheckpointData, TokenUsage


def _make_cp(cp_id: str, workflow_id: str = "wf-integ", wave_index: int = 0) -> CheckpointData:
    task = create_task(title="integ-task", description="integration")
    task = task.model_copy(update={"status": "completed"})
    return CheckpointData(
        id=cp_id,
        workflow_id=workflow_id,
        version=1,
        tasks=[task],
        agent_results={},
        wave_index=wave_index,
        total_token_usage=TokenUsage(),
        created_at=datetime.now(UTC),
    )


@pytest.mark.integration
class TestFilesystemCheckpointStoreIntegration:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, tmp_path: object) -> None:
        store = FilesystemCheckpointStore(str(tmp_path))
        cp1 = _make_cp("cp-1", wave_index=0)
        cp2 = _make_cp("cp-2", wave_index=1)
        cp3 = _make_cp("cp-3", wave_index=2)

        await store.save(cp1)
        await store.save(cp2)
        await store.save(cp3)

        ids = await store.list_checkpoints("wf-integ")
        assert len(ids) == 3

        latest = await store.latest("wf-integ")
        assert latest is not None
        assert latest.id == "cp-3"

        await store.prune("wf-integ", keep_last=1)
        remaining = await store.list_checkpoints("wf-integ")
        assert len(remaining) == 1

        await store.delete(remaining[0])
        assert await store.latest("wf-integ") is None


@pytest.mark.integration
class TestSQLiteCheckpointStoreIntegration:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, tmp_path: object) -> None:
        from anycode.checkpoint.store import SQLiteCheckpointStore

        store = SQLiteCheckpointStore(f"{tmp_path}/checkpoints.db")
        await store.setup()
        try:
            cp1 = _make_cp("cp-1", wave_index=0)
            cp2 = _make_cp("cp-2", wave_index=1)

            await store.save(cp1)
            await store.save(cp2)

            loaded = await store.load("cp-1")
            assert loaded is not None
            assert loaded.wave_index == 0

            latest = await store.latest("wf-integ")
            assert latest is not None
            assert latest.id == "cp-2"

            ids = await store.list_checkpoints("wf-integ")
            assert len(ids) == 2

            await store.prune("wf-integ", keep_last=1)
            ids_after = await store.list_checkpoints("wf-integ")
            assert len(ids_after) == 1
        finally:
            await store.teardown()
