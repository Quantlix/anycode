"""Unit tests for HITL approval gates and manager."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from anycode.hitl.approval import ApprovalManager
from anycode.hitl.channels import CallbackApprovalGate
from anycode.hitl.review import format_approval_request
from anycode.types import ApprovalConfig, ApprovalRequest, ApprovalResponse


def _make_request(
    request_type: str = "tool_call",
    agent: str = "agent-1",
    description: str = "Execute bash: rm -rf /",
    context: dict | None = None,
) -> ApprovalRequest:
    return ApprovalRequest(
        id=f"req-{request_type}",
        type=request_type,  # type: ignore[arg-type]
        agent=agent,
        description=description,
        context=context or {"tool_name": "bash", "command": "rm -rf /"},
        created_at=datetime.now(UTC),
    )


class TestCallbackApprovalGate:
    @pytest.mark.asyncio
    async def test_approve(self) -> None:
        async def always_approve(req: ApprovalRequest) -> ApprovalResponse:
            return ApprovalResponse(approved=True, request_id=req.id, responded_at=datetime.now(UTC))

        gate = CallbackApprovalGate(always_approve)
        resp = await gate.request_approval(_make_request())
        assert resp.approved is True

    @pytest.mark.asyncio
    async def test_reject(self) -> None:
        async def always_reject(req: ApprovalRequest) -> ApprovalResponse:
            return ApprovalResponse(approved=False, reason="dangerous", request_id=req.id, responded_at=datetime.now(UTC))

        gate = CallbackApprovalGate(always_reject)
        resp = await gate.request_approval(_make_request())
        assert resp.approved is False
        assert resp.reason == "dangerous"

    @pytest.mark.asyncio
    async def test_modify_input(self) -> None:
        async def modify_cmd(req: ApprovalRequest) -> ApprovalResponse:
            return ApprovalResponse(approved=True, modified_input={"command": "ls"}, request_id=req.id, responded_at=datetime.now(UTC))

        gate = CallbackApprovalGate(modify_cmd)
        resp = await gate.request_approval(_make_request())
        assert resp.approved is True
        assert resp.modified_input == {"command": "ls"}


class TestApprovalManager:
    @pytest.mark.asyncio
    async def test_disabled_skips_approval(self) -> None:
        config = ApprovalConfig(enabled=False)

        async def should_not_call(req: ApprovalRequest) -> ApprovalResponse:
            raise AssertionError("Should not be called")

        gate = CallbackApprovalGate(should_not_call)
        mgr = ApprovalManager(config, gate)
        resp = await mgr.check_and_request(request_type="tool_call", agent="a1", description="ls")
        assert resp is None

    @pytest.mark.asyncio
    async def test_tool_not_in_require_list_auto_skips(self) -> None:
        config = ApprovalConfig(enabled=True, require_approval_tools=["file_write"])

        async def should_not_call(req: ApprovalRequest) -> ApprovalResponse:
            raise AssertionError("Should not be called")

        gate = CallbackApprovalGate(should_not_call)
        mgr = ApprovalManager(config, gate)
        resp = await mgr.check_and_request(request_type="tool_call", agent="a1", description="bash ls", context={"tool_name": "bash"})
        assert resp is None

    @pytest.mark.asyncio
    async def test_tool_in_require_list_triggers_approval(self) -> None:
        config = ApprovalConfig(enabled=True, require_approval_tools=["bash"])

        async def approve_all(req: ApprovalRequest) -> ApprovalResponse:
            return ApprovalResponse(approved=True, request_id=req.id, responded_at=datetime.now(UTC))

        gate = CallbackApprovalGate(approve_all)
        mgr = ApprovalManager(config, gate)
        resp = await mgr.check_and_request(request_type="tool_call", agent="a1", description="bash", context={"tool_name": "bash"})
        assert resp is not None
        assert resp.approved is True

    @pytest.mark.asyncio
    async def test_task_approval_when_disabled(self) -> None:
        config = ApprovalConfig(enabled=True, require_approval_tasks=False)

        async def should_not_call(req: ApprovalRequest) -> ApprovalResponse:
            raise AssertionError("Should not be called")

        gate = CallbackApprovalGate(should_not_call)
        mgr = ApprovalManager(config, gate)
        resp = await mgr.check_and_request(request_type="task", agent="a1", description="run task")
        assert resp is None

    @pytest.mark.asyncio
    async def test_timeout_returns_default(self) -> None:
        config = ApprovalConfig(enabled=True, timeout_seconds=0.05, require_approval_tools=["bash"])

        async def slow_gate(req: ApprovalRequest) -> ApprovalResponse:
            await asyncio.sleep(5)
            return ApprovalResponse(approved=True, request_id=req.id, responded_at=datetime.now(UTC))

        gate = CallbackApprovalGate(slow_gate)
        mgr = ApprovalManager(config, gate)
        resp = await mgr.check_and_request(request_type="tool_call", agent="a1", description="bash", context={"tool_name": "bash"})
        assert resp is not None
        # default_on_timeout is "reject"
        assert resp.approved is False

    @pytest.mark.asyncio
    async def test_history_tracked(self) -> None:
        config = ApprovalConfig(enabled=True, require_approval_tools=["bash"])

        async def approve_all(req: ApprovalRequest) -> ApprovalResponse:
            return ApprovalResponse(approved=True, request_id=req.id, responded_at=datetime.now(UTC))

        gate = CallbackApprovalGate(approve_all)
        mgr = ApprovalManager(config, gate)
        await mgr.check_and_request(request_type="tool_call", agent="a1", description="ls", context={"tool_name": "bash"})
        await mgr.check_and_request(request_type="tool_call", agent="a1", description="cat", context={"tool_name": "bash"})
        assert len(mgr.history) == 2


class TestFormatApprovalRequest:
    def test_produces_box_output(self) -> None:
        req = _make_request()
        output = format_approval_request(req)
        assert "APPROVAL REQUIRED" in output
        assert "agent-1" in output

    def test_includes_context(self) -> None:
        req = _make_request(context={"tool_name": "bash", "task_id": "my-task"})
        output = format_approval_request(req)
        assert "my-task" in output

    def test_no_context(self) -> None:
        req = _make_request(context=None)
        output = format_approval_request(req)
        assert "APPROVAL REQUIRED" in output
