"""Unit tests for checkpoint serialization, stores, and manager."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from anycode.checkpoint.manager import CheckpointManager
from anycode.checkpoint.serializer import deserialize_checkpoint, serialize_checkpoint
from anycode.checkpoint.store import FilesystemCheckpointStore, SQLiteCheckpointStore
from anycode.tasks.task import create_task
from anycode.types import (
    AgentRunResult,
    CheckpointConfig,
    CheckpointData,
    ImageBlock,
    ImageSource,
    LLMMessage,
    TextBlock,
    TokenUsage,
    ToolCallRecord,
    ToolResultBlock,
    ToolUseBlock,
)


def _make_checkpoint(
    checkpoint_id: str = "cp-001",
    workflow_id: str = "wf-test",
    wave_index: int = 0,
) -> CheckpointData:
    task = create_task(title="Test task", description="A test task")
    task = task.model_copy(update={"status": "completed", "result": "done"})
    messages = [
        LLMMessage(role="user", content=[TextBlock(text="Hello")]),
        LLMMessage(
            role="assistant",
            content=[
                TextBlock(text="Using tool"),
                ToolUseBlock(id="tu-1", name="bash", input={"command": "ls"}),
            ],
        ),
        LLMMessage(role="user", content=[ToolResultBlock(tool_use_id="tu-1", content="file.txt")]),
    ]
    return CheckpointData(
        id=checkpoint_id,
        workflow_id=workflow_id,
        version=1,
        tasks=[task],
        agent_results={
            "agent-1": AgentRunResult(
                success=True,
                output="done",
                messages=messages,
                token_usage=TokenUsage(input_tokens=100, output_tokens=50),
                tool_calls=[ToolCallRecord(tool_name="bash", input={"command": "ls"}, output="file.txt", duration=0.5)],
            )
        },
        wave_index=wave_index,
        total_token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        created_at=datetime.now(UTC),
    )


class TestSerializer:
    def test_round_trip_text_block(self) -> None:
        cp = _make_checkpoint()
        raw = serialize_checkpoint(cp)
        restored = deserialize_checkpoint(raw)
        assert restored.id == cp.id
        assert restored.workflow_id == cp.workflow_id
        assert restored.wave_index == cp.wave_index
        assert len(restored.tasks) == 1
        assert restored.tasks[0].title == "Test task"

    def test_round_trip_tool_blocks(self) -> None:
        cp = _make_checkpoint()
        raw = serialize_checkpoint(cp)
        restored = deserialize_checkpoint(raw)
        agent_result = restored.agent_results["agent-1"]
        assert len(agent_result.messages) == 3
        msg2 = agent_result.messages[1]
        assert len(msg2.content) == 2
        assert isinstance(msg2.content[0], TextBlock)
        assert isinstance(msg2.content[1], ToolUseBlock)
        msg3 = agent_result.messages[2]
        assert isinstance(msg3.content[0], ToolResultBlock)

    def test_round_trip_image_block(self) -> None:
        messages = [LLMMessage(role="user", content=[ImageBlock(source=ImageSource(media_type="image/png", data="aGVsbG8="))])]
        cp = CheckpointData(
            id="cp-img",
            workflow_id="wf-img",
            version=1,
            tasks=[],
            agent_results={"agent-1": AgentRunResult(success=True, output="", messages=messages, token_usage=TokenUsage(), tool_calls=[])},
            wave_index=0,
            total_token_usage=TokenUsage(),
            created_at=datetime.now(UTC),
        )
        raw = serialize_checkpoint(cp)
        restored = deserialize_checkpoint(raw)
        img = restored.agent_results["agent-1"].messages[0].content[0]
        assert isinstance(img, ImageBlock)
        assert img.source.data == "aGVsbG8="

    def test_round_trip_token_usage(self) -> None:
        cp = _make_checkpoint()
        raw = serialize_checkpoint(cp)
        restored = deserialize_checkpoint(raw)
        assert restored.total_token_usage.input_tokens == 100
        assert restored.total_token_usage.output_tokens == 50

    def test_round_trip_tool_calls(self) -> None:
        cp = _make_checkpoint()
        raw = serialize_checkpoint(cp)
        restored = deserialize_checkpoint(raw)
        tc = restored.agent_results["agent-1"].tool_calls
        assert len(tc) == 1
        assert tc[0].tool_name == "bash"
        assert tc[0].duration == 0.5


class TestFilesystemCheckpointStore:
    @pytest.mark.asyncio
    async def test_save_and_load(self, tmp_path: object) -> None:
        store = FilesystemCheckpointStore(str(tmp_path))
        cp = _make_checkpoint()
        await store.save(cp)
        loaded = await store.load("cp-001")
        assert loaded is not None
        assert loaded.id == "cp-001"

    @pytest.mark.asyncio
    async def test_latest(self, tmp_path: object) -> None:
        store = FilesystemCheckpointStore(str(tmp_path))
        await store.save(_make_checkpoint("cp-001", wave_index=0))
        await store.save(_make_checkpoint("cp-002", wave_index=1))
        latest = await store.latest("wf-test")
        assert latest is not None
        assert latest.id == "cp-002"

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, tmp_path: object) -> None:
        store = FilesystemCheckpointStore(str(tmp_path))
        await store.save(_make_checkpoint("cp-001"))
        await store.save(_make_checkpoint("cp-002"))
        ids = await store.list_checkpoints("wf-test")
        assert len(ids) == 2

    @pytest.mark.asyncio
    async def test_prune(self, tmp_path: object) -> None:
        store = FilesystemCheckpointStore(str(tmp_path))
        for i in range(5):
            await store.save(_make_checkpoint(f"cp-{i:03d}"))
        await store.prune("wf-test", keep_last=2)
        ids = await store.list_checkpoints("wf-test")
        assert len(ids) == 2

    @pytest.mark.asyncio
    async def test_delete(self, tmp_path: object) -> None:
        store = FilesystemCheckpointStore(str(tmp_path))
        await store.save(_make_checkpoint("cp-001"))
        await store.delete("cp-001")
        loaded = await store.load("cp-001")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, tmp_path: object) -> None:
        store = FilesystemCheckpointStore(str(tmp_path))
        assert await store.load("missing") is None

    @pytest.mark.asyncio
    async def test_latest_empty_workflow(self, tmp_path: object) -> None:
        store = FilesystemCheckpointStore(str(tmp_path))
        assert await store.latest("no-such-workflow") is None


class TestSQLiteCheckpointStore:
    @pytest.mark.asyncio
    async def test_save_and_load(self) -> None:

        store = SQLiteCheckpointStore(":memory:")
        await store.setup()
        try:
            cp = _make_checkpoint()
            await store.save(cp)
            loaded = await store.load("cp-001")
            assert loaded is not None
            assert loaded.id == "cp-001"
        finally:
            await store.teardown()

    @pytest.mark.asyncio
    async def test_latest(self) -> None:

        store = SQLiteCheckpointStore(":memory:")
        await store.setup()
        try:
            await store.save(_make_checkpoint("cp-001", wave_index=0))
            await store.save(_make_checkpoint("cp-002", wave_index=1))
            latest = await store.latest("wf-test")
            assert latest is not None
            assert latest.id == "cp-002"
        finally:
            await store.teardown()

    @pytest.mark.asyncio
    async def test_prune(self) -> None:

        store = SQLiteCheckpointStore(":memory:")
        await store.setup()
        try:
            for i in range(5):
                await store.save(_make_checkpoint(f"cp-{i:03d}"))
            await store.prune("wf-test", keep_last=2)
            ids = await store.list_checkpoints("wf-test")
            assert len(ids) == 2
        finally:
            await store.teardown()


class TestCheckpointManager:
    @pytest.mark.asyncio
    async def test_auto_save_and_load_latest(self, tmp_path: object) -> None:
        config = CheckpointConfig(enabled=True, path=str(tmp_path), keep_last=3)
        mgr = CheckpointManager(config)
        task = create_task(title="t1", description="d1")
        cp = await mgr.auto_save("wf-1", [task], {}, wave_index=0, total_usage=TokenUsage(input_tokens=10, output_tokens=5))
        loaded = await mgr.load_latest("wf-1")
        assert loaded is not None
        assert loaded.id == cp.id

    def test_spec_change_detection(self) -> None:
        config = CheckpointConfig(enabled=True)
        mgr = CheckpointManager(config)
        t1 = create_task(title="task-A", description="desc-A")
        t2 = create_task(title="task-A", description="desc-A")
        t3 = create_task(title="task-B", description="desc-B")
        cp = _make_checkpoint()
        cp_same = cp.model_copy(update={"tasks": [t1]})
        assert not mgr.detect_spec_change([t2], cp_same)
        cp_diff = cp.model_copy(update={"tasks": [t3]})
        assert mgr.detect_spec_change([t1], cp_diff)

    def test_compute_spec_hash_deterministic(self) -> None:
        t1 = create_task(title="A", description="B")
        t2 = create_task(title="A", description="B")
        assert CheckpointManager.compute_spec_hash([t1]) == CheckpointManager.compute_spec_hash([t2])
