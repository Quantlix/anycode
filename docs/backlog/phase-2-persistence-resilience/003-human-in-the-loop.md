# 2.3 Human-in-the-Loop Approval Gates

> **Priority:** Medium
> **Complexity:** Medium
> **Status:** Complete
> **Phase:** 2 — Persistence & Resilience

## Problem

Enterprises cannot deploy fully autonomous agents for financial, legal, or security-sensitive workflows without human checkpoints. Currently, AnyCode has no way to pause execution and wait for human input before proceeding.

Use cases:
- Approve a financial transaction before an agent executes it
- Review generated code before an agent deploys it
- Validate a plan before agents start implementing it
- Confirm deletion of resources before an agent proceeds

## Solution

Introduce approval gates that can be attached to tasks, tool calls, or specific agent turns. When an approval gate is triggered, execution pauses and waits for human input via one of several configurable channels.

## Files to Create

```
src/anycode/hitl/
├── __init__.py
├── approval.py      # ApprovalGate Protocol + implementations
├── review.py        # Present outputs for review before proceeding
└── channels.py      # Callback, stdin, webhook approval channels
```

## Files to Modify

| File | Change |
|------|--------|
| `src/anycode/types.py` | Add `ApprovalRequest`, `ApprovalResponse`, `ApprovalGate` Protocol |
| `src/anycode/core/orchestrator.py` | Check approval requirement before executing tasks |
| `src/anycode/tools/executor.py` | Check approval requirement before executing sensitive tools |

## New Types

```python
class ApprovalRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    type: Literal["task", "tool_call", "output"]
    agent: str
    description: str
    context: dict[str, Any] | None = None
    created_at: datetime


class ApprovalResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    request_id: str
    approved: bool
    reason: str | None = None
    modified_input: dict[str, Any] | None = None  # Allow human to modify before proceeding
    responded_at: datetime


@runtime_checkable
class ApprovalGate(Protocol):
    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse: ...


class ApprovalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool = False
    timeout_seconds: float = 300.0          # 5 minute default timeout
    default_on_timeout: Literal["approve", "reject"] = "reject"
    require_approval_tools: list[str] | None = None  # Tool names requiring approval
    require_approval_tasks: bool = False              # All tasks require approval
```

## API Design

### Callback-based (programmatic):

```python
async def my_approval_handler(request: ApprovalRequest) -> ApprovalResponse:
    # Custom logic — could call a Slack bot, send email, etc.
    if request.type == "tool_call" and "bash" in request.description:
        return ApprovalResponse(
            request_id=request.id,
            approved=False,
            reason="Shell commands require manual review",
            responded_at=datetime.now(UTC),
        )
    return ApprovalResponse(
        request_id=request.id,
        approved=True,
        responded_at=datetime.now(UTC),
    )

engine = AnyCode(config={
    "approval": ApprovalConfig(
        enabled=True,
        require_approval_tools=["bash", "file_write"],
    ),
    "approval_handler": my_approval_handler,
})
```

### Interactive stdin (CLI):

```python
from anycode.hitl import StdinApprovalGate

engine = AnyCode(config={
    "approval": ApprovalConfig(enabled=True),
    "approval_handler": StdinApprovalGate(),
})

# During execution, prints to console:
# ┌─────────────────────────────────────────────────┐
# │ APPROVAL REQUIRED                                │
# │ Agent: builder                                   │
# │ Action: Execute bash command                     │
# │ Command: rm -rf /tmp/old-project                 │
# │                                                  │
# │ [a]pprove  [r]eject  [m]odify                   │
# └─────────────────────────────────────────────────┘
```

### Webhook-based (async):

```python
from anycode.hitl import WebhookApprovalGate

gate = WebhookApprovalGate(
    request_url="https://api.mycompany.com/approvals",
    poll_url="https://api.mycompany.com/approvals/{request_id}/status",
    poll_interval=5.0,
)
```

## Approval Flow

```
Task/tool execution triggers approval:
  1. Build ApprovalRequest with context
  2. Check if approval is required (config + tool name match)
  3. If required:
     a. Emit "approval:requested" event
     b. Call approval_handler.request_approval(request)
     c. Wait with timeout
     d. On timeout → use default_on_timeout
     e. On approval → proceed with execution
     f. On rejection → skip with reason in result
     g. On modified_input → re-execute with modified parameters
  4. If not required → proceed normally
```

## Implementation Notes

- Approval gates are async — they don't block the event loop
- Timeout is enforced via `asyncio.wait_for()`
- `StdinApprovalGate` uses `asyncio.get_event_loop().run_in_executor()` for non-blocking stdin
- Approval events are emitted on the team event bus for logging/monitoring
- The `modified_input` field allows humans to adjust parameters before proceeding
- Approval history is recorded for audit trails
- Works with checkpointing — if process restarts during an approval wait, the task is re-queued

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Blocking on stdin in non-interactive environments | Medium | Detect non-interactive terminal; use callback mode |
| Timeout too short for complex approvals | Low | Configurable timeout; "reject" is safe default |
| Webhook endpoint unavailable | Medium | Retry with backoff; fallback to reject |
| Approval state lost on crash | Medium | Integrate with checkpoint system |

## Tests

**File:** `tests/test_hitl.py`

- [ ] Callback approval gate approves and execution proceeds
- [ ] Callback approval gate rejects and execution skips
- [ ] Timeout triggers default action (approve or reject)
- [ ] Modified input is used for re-execution
- [ ] Approval is required only for specified tools
- [ ] Approval events are emitted on event bus
- [ ] StdinApprovalGate formats request correctly
- [ ] WebhookApprovalGate sends request and polls for response
- [ ] Disabled approval config skips all approval checks
- [ ] Approval history is recorded

## Estimated Effort

Medium — 3-4 working sessions
