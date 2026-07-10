---
title: "Add Human-in-the-Loop Approval Gates to AnyCode"
description: "Require a human to approve sensitive AnyCode tool calls or tasks with ApprovalManager and approval gates for stdin, callbacks, or webhooks, including timeout policy."
keywords: anycode human in the loop, HITL approval, ApprovalManager, ApprovalConfig, approval gate, StdinApprovalGate, WebhookApprovalGate, approve tool call, require_approval_tools
---

# Human-in-the-Loop Approval

Some actions should never run unsupervised — deleting data, sending a message, opening a pull request, spending money. An approval gate pauses the workflow and waits for a person to allow or deny the action. This guide wires up `ApprovalManager`, the built-in gates, and the timeout policy that decides what happens when nobody answers.

## When to require approval

Reach for a gate whenever an action crosses a trust boundary or is hard to undo:

- Irreversible operations — deleting records, force-pushing, publishing.
- External side effects — sending email, posting to Slack, calling a paid API.
- Anything touching private or regulated data.

## Wire up an approval manager

`ApprovalManager` pairs an `ApprovalConfig` (what needs approval) with an `ApprovalGate` (how a human answers). The simplest gate wraps your own async function.

```python title="approval.py"
from datetime import UTC, datetime

from anycode import ApprovalManager, CallbackApprovalGate
from anycode.types import ApprovalConfig, ApprovalRequest, ApprovalResponse


async def review(request: ApprovalRequest) -> ApprovalResponse:
    # Show request.description to a human; here we auto-approve for illustration.
    return ApprovalResponse(approved=True, request_id=request.id, responded_at=datetime.now(UTC))


config = ApprovalConfig(enabled=True, require_approval_tools=["bash", "file_write"])
manager = ApprovalManager(config, CallbackApprovalGate(review))

decision = await manager.check_and_request(
    request_type="tool_call",
    agent="worker",
    description="Run: bash rm -rf ./build",
    context={"tool_name": "bash"},
)
```

`check_and_request` returns an `ApprovalResponse` when a human was consulted, or `None` when approval was **not required** for that action.

!!! warning "`None` means 'not required', not 'rejected'"
    Treat the return value carefully: `None` is a skip (the action proceeds), while an `ApprovalResponse` with `approved=False` is an explicit rejection. And `require_approval_tools=None` does **not** mean "approve nothing" — it bypasses the per-tool filter so *every* tool call is sent to the gate. For tool-call filtering to work, `context` must include `tool_name`.

## Configure what needs approval

| `ApprovalConfig` field | Default | Effect |
| --- | --- | --- |
| `enabled` | `False` | Master switch — off means every check returns `None` |
| `require_approval_tools` | `None` | Only these tool names need approval (`None` = all reach the gate) |
| `require_approval_tasks` | `False` | Require approval before each task runs |
| `timeout_seconds` | `300.0` | How long to wait for a human |
| `default_on_timeout` | `"reject"` | What to decide when the wait elapses |

The timeout policy is the safety net: if no one answers within `timeout_seconds`, the manager auto-decides based on `default_on_timeout`. Keep it `"reject"` for anything dangerous.

## Choose a gate

| Gate | How a human answers |
| --- | --- |
| `CallbackApprovalGate(handler)` | Your async function — route to any UI, queue, or chatbot |
| `StdinApprovalGate()` | Interactive terminal prompt (`[a]pprove / [r]eject / [m]odify`) |
| `WebhookApprovalGate(request_url, poll_url)` | POSTs the request, polls a URL until a decision lands |

`StdinApprovalGate` requires a real TTY and raises otherwise, so use `CallbackApprovalGate` or `WebhookApprovalGate` for services and CI.

## Turn it on for a whole engine

To gate a team run, set both the config **and** the handler on `OrchestratorConfig`:

```python title="engine.py"
from anycode import AnyCode
from anycode.types import ApprovalConfig

engine = AnyCode(config={
    "approval": ApprovalConfig(enabled=True, require_approval_tools=["bash"]),
    "approval_handler": CallbackApprovalGate(review),
})
```

!!! danger "No handler, no gate"
    If `approval_handler` is `None`, no approval manager is created even when `enabled=True`. There is no default gate — you must supply one.

## Next steps

- [Build a support-triage system](../tutorials/support-triage.md) — approval gates guarding a refund action.
- [Production controls](production-controls.md) — approval alongside budgets and verification.
- [Work with tools](tools.md) — scope which tools an agent can even attempt.
- [Configuration reference](../reference/configuration.md) — every `ApprovalConfig` field.
