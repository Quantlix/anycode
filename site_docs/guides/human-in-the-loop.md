---
title: "Add Human-in-the-Loop Approval Gates to AnyCode"
description: "Require human approval for sensitive AnyCode tools or tasks with stdin, callback, or webhook gates, audit history, timeout policy, and explicit denial."
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

## The complete, runnable program

Here is one whole file that puts the pieces together: an `ApprovalConfig` that gates only `bash` and `file_write`, a `CallbackApprovalGate` whose handler rejects shell commands, and a loop that shows all three outcomes — approved, rejected, and skipped (`None`, not required) — then prints the audit trail. It uses callback gates only, so it runs with no TTY, no API key, and no network.

```python title="approval.py"
import asyncio
from datetime import UTC, datetime

from anycode import ApprovalManager, CallbackApprovalGate
from anycode.types import ApprovalConfig, ApprovalRequest, ApprovalResponse


async def review(request: ApprovalRequest) -> ApprovalResponse:
    """Decide each request. In production, route this to a UI, queue, or chatbot."""
    tool_name = (request.context or {}).get("tool_name", "")
    if tool_name == "bash":
        return ApprovalResponse(
            approved=False,
            reason="Shell commands require manual review",
            request_id=request.id,
            responded_at=datetime.now(UTC),
        )
    return ApprovalResponse(approved=True, request_id=request.id, responded_at=datetime.now(UTC))


async def main() -> None:
    config = ApprovalConfig(
        enabled=True,
        require_approval_tools=["bash", "file_write"],
        timeout_seconds=5.0,
        default_on_timeout="reject",
    )
    manager = ApprovalManager(config, CallbackApprovalGate(review))

    attempts = [
        ("bash", "Run: rm -rf ./build"),
        ("file_write", "Write report to ./out/report.md"),
        ("file_read", "Read ./src/app.py"),
    ]
    for tool_name, description in attempts:
        decision = await manager.check_and_request(
            request_type="tool_call",
            agent="worker",
            description=description,
            context={"tool_name": tool_name},
        )
        if decision is None:
            print(f"[skipped ] {tool_name}: approval not required")
        elif decision.approved:
            print(f"[approved] {tool_name}: {description}")
        else:
            print(f"[rejected] {tool_name}: {decision.reason}")

    print("\n--- audit trail ---")
    for request, response in manager.history:
        verdict = "APPROVED" if response.approved else f"REJECTED ({response.reason})"
        print(f"  {request.description:34s} -> {verdict}")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it from the project root:

```bash
uv run python approval.py
```

!!! tip "Tested copy"
    See [`examples/08_hitl_approval.py`](https://github.com/Quantlix/anycode/blob/main/examples/08_hitl_approval.py) for the CI-tested version, which also covers timeout handling, modified-input flows, task-level gates, and console rendering of a request.

## Next steps

- [Build a support-triage system](../tutorials/support-triage.md) — approval gates guarding a refund action.
- [Production controls](production-controls.md) — approval alongside budgets and verification.
- [Work with tools](tools.md) — scope which tools an agent can even attempt.
- [Configuration reference](../reference/configuration.md) — every `ApprovalConfig` field.
