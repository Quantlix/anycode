"""Integration test: full pipeline with memory + checkpoint + HITL."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from anycode.checkpoint.manager import CheckpointManager
from anycode.hitl.approval import ApprovalManager
from anycode.hitl.channels import CallbackApprovalGate
from anycode.memory.composite import CompositeMemory
from anycode.memory.vector_store import InMemoryVectorStore
from anycode.tasks.task import create_task
from anycode.types import (
    ApprovalConfig,
    ApprovalRequest,
    ApprovalResponse,
    CheckpointConfig,
    TokenUsage,
)


@pytest.mark.integration
class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_memory_checkpoint_hitl_roundtrip(self, tmp_path: object) -> None:
        """End-to-end: store memory, create checkpoint, run approval gate."""
        from anycode.memory.sqlite_store import SQLiteStore

        # -- Memory --
        kv = SQLiteStore(f"{tmp_path}/pipeline.db")
        await kv.setup()
        vector = InMemoryVectorStore()
        memory = CompositeMemory(kv_store=kv, vector_store=vector, auto_index=True)

        await memory.set("context", "User prefers Python for backend")
        got = await memory.get("context")
        assert got is not None

        results = await memory.search("Python backend", top_k=1)
        assert len(results) == 1

        # -- Checkpoint --
        cp_config = CheckpointConfig(enabled=True, path=f"{tmp_path}/checkpoints")
        mgr = CheckpointManager(cp_config)

        task = create_task(title="Implement API", description="Build REST API")
        task = task.model_copy(update={"status": "completed"})

        cp = await mgr.auto_save("wf-pipeline", [task], {}, wave_index=0, total_usage=TokenUsage(input_tokens=50, output_tokens=25))
        loaded = await mgr.load_latest("wf-pipeline")
        assert loaded is not None
        assert loaded.id == cp.id

        # -- HITL --
        approval_log: list[str] = []

        async def gate_fn(req: ApprovalRequest) -> ApprovalResponse:
            approval_log.append(req.type)
            return ApprovalResponse(approved=True, request_id=req.id, responded_at=datetime.now(UTC))

        approval_config = ApprovalConfig(enabled=True, require_approval_tools=["bash"])
        gate = CallbackApprovalGate(gate_fn)
        approval_mgr = ApprovalManager(approval_config, gate)

        resp = await approval_mgr.check_and_request(request_type="tool_call", agent="a1", description="deploy", context={"tool_name": "bash"})
        assert resp is not None
        assert resp.approved is True
        assert approval_log == ["tool_call"]

        # Non-gated tool should be skipped (returns None)
        resp2 = await approval_mgr.check_and_request(request_type="tool_call", agent="a1", description="read", context={"tool_name": "file_read"})
        assert resp2 is None
        assert len(approval_log) == 1  # gate not called for file_read

        await kv.teardown()
