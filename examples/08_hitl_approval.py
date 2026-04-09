# Demo 08 — Human-in-the-Loop Approval Gates
# Execute: uv run python examples/08_hitl_approval.py
#
# Demonstrates:
#   1. CallbackApprovalGate — programmatic approve/reject/modify
#   2. ApprovalManager — config-driven approval enforcement
#   3. Timeout handling — default action on approval timeout
#   4. Approval history — audit trail of all requests/responses
#   5. Tool-specific approval — only require approval for certain tools
#   6. Task approval gates — require approval for task execution
#   7. Format rendering — console box rendering of approval requests
#   8. Modified input flow — human modifies parameters before approval
#
# No external services required (uses callback-based gates, no stdin).

import asyncio
import sys
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

from anycode.hitl.approval import ApprovalManager
from anycode.hitl.channels import CallbackApprovalGate
from anycode.hitl.review import format_approval_request
from anycode.types import ApprovalConfig, ApprovalRequest, ApprovalResponse

SEPARATOR = "=" * 60
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "artifacts"


# --- Helpers ---


def _make_request(
    request_type: str = "tool_call",
    agent: str = "worker-1",
    description: str = "Execute bash command",
    context: dict | None = None,
) -> ApprovalRequest:
    return ApprovalRequest(
        id=f"req-{request_type}-{datetime.now(UTC).strftime('%H%M%S%f')[:10]}",
        type=request_type,  # type: ignore[arg-type]
        agent=agent,
        description=description,
        context=context or {"tool_name": "bash", "command": "ls -la"},
        created_at=datetime.now(UTC),
    )


# --- Section 1: CallbackApprovalGate ---


async def demo_callback_gate() -> None:
    print(f"\n{SEPARATOR}")
    print("  1. CallbackApprovalGate (Programmatic)")
    print(SEPARATOR)

    # --- 1a. Always approve ---
    print("\n--- Always Approve ---")

    async def always_approve(req: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(approved=True, request_id=req.id, responded_at=datetime.now(UTC))

    gate = CallbackApprovalGate(always_approve)
    resp = await gate.request_approval(_make_request())
    print(f"Request: {resp.request_id}")
    print(f"Approved: {resp.approved}")
    print(f"Reason: {resp.reason}")

    # --- 1b. Always reject ---
    print("\n--- Always Reject ---")

    async def always_reject(req: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(approved=False, reason="Dangerous operation", request_id=req.id, responded_at=datetime.now(UTC))

    gate_reject = CallbackApprovalGate(always_reject)
    resp2 = await gate_reject.request_approval(_make_request(description="rm -rf /"))
    print(f"Approved: {resp2.approved}")
    print(f"Reason: {resp2.reason}")

    # --- 1c. Conditional approval ---
    print("\n--- Conditional Approval (by tool name) ---")

    async def conditional_handler(req: ApprovalRequest) -> ApprovalResponse:
        tool_name = (req.context or {}).get("tool_name", "")
        if tool_name == "bash":
            return ApprovalResponse(approved=False, reason="Shell commands require manual review", request_id=req.id, responded_at=datetime.now(UTC))
        return ApprovalResponse(approved=True, request_id=req.id, responded_at=datetime.now(UTC))

    gate_cond = CallbackApprovalGate(conditional_handler)

    bash_req = _make_request(context={"tool_name": "bash", "command": "rm -rf /"})
    read_req = _make_request(description="Read file", context={"tool_name": "file_read", "path": "/src/app.py"})

    resp_bash = await gate_cond.request_approval(bash_req)
    resp_read = await gate_cond.request_approval(read_req)
    print(f"bash:      approved={resp_bash.approved}, reason={resp_bash.reason}")
    print(f"file_read: approved={resp_read.approved}, reason={resp_read.reason}")

    # --- 1d. Modified input ---
    print("\n--- Modified Input ---")

    async def modify_handler(req: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(approved=True, modified_input={"command": "ls -la /tmp"}, request_id=req.id, responded_at=datetime.now(UTC))

    gate_mod = CallbackApprovalGate(modify_handler)
    resp_mod = await gate_mod.request_approval(_make_request(context={"tool_name": "bash", "command": "rm -rf /"}))
    print(f"Approved: {resp_mod.approved}")
    print(f"Modified input: {resp_mod.modified_input}")


# --- Section 2: ApprovalManager ---


async def demo_approval_manager() -> None:
    print(f"\n{SEPARATOR}")
    print("  2. ApprovalManager (Config-Driven)")
    print(SEPARATOR)

    # --- 2a. Disabled config skips everything ---
    print("\n--- Disabled Config ---")

    async def should_not_call(req: ApprovalRequest) -> ApprovalResponse:
        raise AssertionError("Should never be called")

    config_disabled = ApprovalConfig(enabled=False)
    gate = CallbackApprovalGate(should_not_call)
    mgr = ApprovalManager(config_disabled, gate)
    resp = await mgr.check_and_request(request_type="tool_call", agent="a1", description="anything")
    print(f"Disabled config → response: {resp} (None = skipped)  ✓")

    # --- 2b. Tool-specific approval ---
    print("\n--- Tool-Specific Approval ---")

    async def approve_all(req: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(approved=True, request_id=req.id, responded_at=datetime.now(UTC))

    config_tools = ApprovalConfig(enabled=True, require_approval_tools=["bash", "file_write"])
    gate_tools = CallbackApprovalGate(approve_all)
    mgr_tools = ApprovalManager(config_tools, gate_tools)

    # bash → requires approval
    resp1 = await mgr_tools.check_and_request(request_type="tool_call", agent="worker", description="bash ls", context={"tool_name": "bash"})
    print(f"bash → approval required: {resp1 is not None}, approved: {resp1.approved if resp1 else 'N/A'}")

    # file_read → NOT in require list → skipped
    resp2 = await mgr_tools.check_and_request(request_type="tool_call", agent="worker", description="file_read", context={"tool_name": "file_read"})
    print(f"file_read → approval required: {resp2 is not None}")

    # file_write → requires approval
    resp3 = await mgr_tools.check_and_request(request_type="tool_call", agent="writer", description="write file", context={"tool_name": "file_write"})
    print(f"file_write → approval required: {resp3 is not None}, approved: {resp3.approved if resp3 else 'N/A'}")


# --- Section 3: Timeout Handling ---


async def demo_timeout() -> None:
    print(f"\n{SEPARATOR}")
    print("  3. Timeout Handling")
    print(SEPARATOR)

    # --- 3a. Timeout → reject (default) ---
    print("\n--- Timeout → Reject (default) ---")

    async def slow_gate(req: ApprovalRequest) -> ApprovalResponse:
        await asyncio.sleep(10)  # Will be cut off by timeout
        return ApprovalResponse(approved=True, request_id=req.id, responded_at=datetime.now(UTC))

    config_reject = ApprovalConfig(enabled=True, timeout_seconds=0.1, default_on_timeout="reject", require_approval_tools=["bash"])
    gate = CallbackApprovalGate(slow_gate)
    mgr = ApprovalManager(config_reject, gate)

    resp = await mgr.check_and_request(request_type="tool_call", agent="a1", description="bash", context={"tool_name": "bash"})
    assert resp is not None
    print(f"Approved: {resp.approved}")
    print(f"Reason: {resp.reason}")

    # --- 3b. Timeout → approve ---
    print("\n--- Timeout → Approve ---")
    config_approve = ApprovalConfig(enabled=True, timeout_seconds=0.1, default_on_timeout="approve", require_approval_tools=["bash"])
    mgr2 = ApprovalManager(config_approve, CallbackApprovalGate(slow_gate))

    resp2 = await mgr2.check_and_request(request_type="tool_call", agent="a1", description="bash", context={"tool_name": "bash"})
    assert resp2 is not None
    print(f"Approved: {resp2.approved}")
    print(f"Reason: {resp2.reason}")


# --- Section 4: Approval History ---


async def demo_history() -> None:
    print(f"\n{SEPARATOR}")
    print("  4. Approval History (Audit Trail)")
    print(SEPARATOR)

    async def smart_handler(req: ApprovalRequest) -> ApprovalResponse:
        tool = (req.context or {}).get("tool_name", "")
        if tool == "bash":
            return ApprovalResponse(approved=False, reason="Blocked for safety", request_id=req.id, responded_at=datetime.now(UTC))
        return ApprovalResponse(approved=True, request_id=req.id, responded_at=datetime.now(UTC))

    config = ApprovalConfig(enabled=True, require_approval_tools=["bash", "file_write", "file_read"])
    gate = CallbackApprovalGate(smart_handler)
    mgr = ApprovalManager(config, gate)

    # Generate several approval requests
    tools = [
        ("bash", "Execute shell command"),
        ("file_write", "Write to /tmp/output.txt"),
        ("file_read", "Read /src/config.py"),
        ("bash", "Run tests: pytest"),
    ]
    for tool_name, desc in tools:
        await mgr.check_and_request(request_type="tool_call", agent="worker", description=desc, context={"tool_name": tool_name})

    # Print audit trail
    print(f"\nTotal approvals: {len(mgr.history)}")
    print("\n--- Audit Trail ---")
    for req, resp in mgr.history:
        status = "✓ APPROVED" if resp.approved else f"✗ REJECTED ({resp.reason})"
        print(f"  [{req.created_at.strftime('%H:%M:%S')}] {req.description:30s} → {status}")


# --- Section 5: Task Approval ---


async def demo_task_approval() -> None:
    print(f"\n{SEPARATOR}")
    print("  5. Task Approval Gates")
    print(SEPARATOR)

    # --- Tasks not requiring approval ---
    print("\n--- Tasks Not Requiring Approval ---")

    async def should_not_call(req: ApprovalRequest) -> ApprovalResponse:
        raise AssertionError("Should never be called")

    config_no_task = ApprovalConfig(enabled=True, require_approval_tasks=False, require_approval_tools=["bash"])
    mgr = ApprovalManager(config_no_task, CallbackApprovalGate(should_not_call))
    resp = await mgr.check_and_request(request_type="task", agent="a1", description="Build project")
    print(f"Task approval required: {resp is not None}")

    # --- Tasks requiring approval ---
    print("\n--- Tasks Requiring Approval ---")

    async def task_reviewer(req: ApprovalRequest) -> ApprovalResponse:
        if "deploy" in req.description.lower():
            return ApprovalResponse(
                approved=False,
                reason="Deployment requires team lead approval",
                request_id=req.id,
                responded_at=datetime.now(UTC),
            )
        return ApprovalResponse(approved=True, request_id=req.id, responded_at=datetime.now(UTC))

    config_task = ApprovalConfig(enabled=True, require_approval_tasks=True)
    mgr_task = ApprovalManager(config_task, CallbackApprovalGate(task_reviewer))

    tasks_to_check = [
        "Analyze codebase",
        "Generate unit tests",
        "Deploy to production",
        "Write documentation",
    ]
    for desc in tasks_to_check:
        resp = await mgr_task.check_and_request(request_type="task", agent="orchestrator", description=desc)
        assert resp is not None
        status = "APPROVED" if resp.approved else f"REJECTED ({resp.reason})"
        print(f"  {desc:30s} → {status}")


# --- Section 6: Console Rendering ---


async def demo_console_rendering() -> None:
    print(f"\n{SEPARATOR}")
    print("  6. Console Rendering")
    print(SEPARATOR)

    # Simple request
    print("\n--- Simple Request ---")
    req1 = _make_request(description="Execute: rm -rf /tmp/old-project")
    print(format_approval_request(req1))

    # Request with rich context
    print("\n--- Request with Rich Context ---")
    req2 = _make_request(
        request_type="task",
        agent="deployer",
        description="Deploy v2.0 to production",
        context={"environment": "production", "version": "2.0.0", "rollback_plan": "Revert to v1.9.3 if errors exceed 1%"},
    )
    print(format_approval_request(req2))

    # Request with no context
    print("\n--- Request with No Context ---")
    req3 = ApprovalRequest(
        id="req-bare",
        type="output",
        agent="writer",
        description="Publish generated report",
        context=None,
        created_at=datetime.now(UTC),
    )
    print(format_approval_request(req3))


# --- Section 7: Modified Input Flow ---


async def demo_modified_input_flow() -> None:
    print(f"\n{SEPARATOR}")
    print("  7. Modified Input Flow")
    print(SEPARATOR)

    print("\n--- Simulated Modification Pipeline ---")

    async def safety_modifier(req: ApprovalRequest) -> ApprovalResponse:
        ctx = req.context or {}
        tool = ctx.get("tool_name", "")
        command = ctx.get("command", "")

        # Simulate human modifying dangerous commands
        if tool == "bash" and "rm" in str(command):
            safe_command = str(command).replace("rm -rf /", "ls /tmp")
            return ApprovalResponse(
                approved=True,
                modified_input={"command": safe_command},
                reason="Modified: replaced destructive command with safe alternative",
                request_id=req.id,
                responded_at=datetime.now(UTC),
            )
        return ApprovalResponse(approved=True, request_id=req.id, responded_at=datetime.now(UTC))

    config = ApprovalConfig(enabled=True, require_approval_tools=["bash", "file_write"])
    mgr = ApprovalManager(config, CallbackApprovalGate(safety_modifier))

    # Dangerous command → gets modified
    resp1 = await mgr.check_and_request(
        request_type="tool_call", agent="builder", description="Execute shell", context={"tool_name": "bash", "command": "rm -rf /"}
    )
    assert resp1 is not None
    print("Original command: rm -rf /")
    print(f"Approved: {resp1.approved}")
    print(f"Modified input: {resp1.modified_input}")
    print(f"Reason: {resp1.reason}")

    # Safe command → passes through
    resp2 = await mgr.check_and_request(
        request_type="tool_call", agent="builder", description="Execute shell", context={"tool_name": "bash", "command": "echo hello"}
    )
    assert resp2 is not None
    print("\nOriginal command: echo hello")
    print(f"Approved: {resp2.approved}")
    print(f"Modified input: {resp2.modified_input}")

    # Full history
    print("\n--- Approval History ---")
    for req, resp in mgr.history:
        modified = f" → modified to {resp.modified_input}" if resp.modified_input else ""
        print(f"  {req.description}: approved={resp.approved}{modified}")


# --- Main ---


class _OutputCapture:
    def __init__(self) -> None:
        self._buffer = StringIO()
        self._stdout = sys.stdout

    def write(self, text: str) -> None:
        self._stdout.write(text)
        self._buffer.write(text)

    def flush(self) -> None:
        self._stdout.flush()

    def get_output(self) -> str:
        return self._buffer.getvalue()


async def main() -> None:
    capture = _OutputCapture()
    sys.stdout = capture  # type: ignore[assignment]

    print("AnyCode — Human-in-the-Loop Approval Gates Demo")

    await demo_callback_gate()
    await demo_approval_manager()
    await demo_timeout()
    await demo_history()
    await demo_task_approval()
    await demo_console_rendering()
    await demo_modified_input_flow()

    print(f"\n{SEPARATOR}")
    print("  All HITL approval demos completed successfully!")
    print(SEPARATOR)

    sys.stdout = capture._stdout
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "08-hitl-approval-output.txt"
    output_file.write_text(capture.get_output(), encoding="utf-8")
    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
