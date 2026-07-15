---
title: "Run AnyCode Work in Sandbox Providers"
description: Configure Daytona or companion sandbox providers with immutable identity, network policy, secret references, command streaming, files, cancellation, and evidence.
keywords: AnyCode sandbox, Daytona Python sandbox, AI agent code execution, policy sandbox provider, isolated agent tools
---

# Run work through sandbox providers

The `SandboxProvider` protocol gives AnyCode one explicit boundary for remote execution, files, streaming, cancellation, cleanup, health, and evidence. Use `DaytonaSandboxProvider` for the maintained Daytona SDK or `CompanionSandboxAdapter` for a separately deployed sandbox service. Wrap either provider with `PolicySandboxProvider` when every operation needs an external authorization decision.

## Install the Daytona integration

```bash
uv add "anycode-py[sandbox]"
```

Configure Daytona credentials through the SDK's supported host configuration. Do not place credentials in `SandboxSpec`, a command environment, labels, logs, or durable run data.

## Create and use a Daytona sandbox

```python
from anycode import (
    DaytonaSandboxProvider,
    ExecutionContext,
    SandboxCommand,
    SandboxSpec,
)

provider = DaytonaSandboxProvider()
spec = SandboxSpec(
    run_id="run-42",
    task_id="task-check",
    correlation_id="correlation-42",
    context=ExecutionContext(
        principal="user:reviewer-42",
        tenant_scope="tenant:example",
        classification="internal",
    ),
    image="python:3.12-slim",
    network="allowlist",
    allowed_domains=("pypi.org", "files.pythonhosted.org"),
    secret_references={"PACKAGE_TOKEN": "daytona:package-token"},
    labels={"workload": "dependency-check"},
)

created = await provider.create(spec)
if not created.ok or created.handle is None:
    raise RuntimeError(created.error.message if created.error else "create failed")

handle = created.handle
try:
    result = await provider.execute(
        handle,
        SandboxCommand(command="python --version", timeout_seconds=30),
    )
    if not result.ok:
        raise RuntimeError(result.error.message if result.error else result.stderr)
    print(result.stdout, result.evidence.digest if result.evidence else "")
finally:
    await provider.destroy(handle)
```

Choose either `image` or `snapshot`, never both. The Daytona adapter currently reports point-in-time snapshots as unsupported, so prefer an image unless a future capability report explicitly confirms snapshot support.

## Apply network and secret rules

| Setting | Behavior |
| --- | --- |
| `network="none"` | No allowlist entries may be supplied |
| `network="allowlist"` | At least one domain or CIDR is required |
| `network="unrestricted"` | The provider may allow unrestricted egress; use only under reviewed host policy |
| `secret_references` | Every value must start with `daytona:` |
| `SandboxCommand.environment` | Credential-like variable names are rejected |

The provider and host must enforce the requested network policy. The model validates the request, but that validation is not proof of kernel, container, VM, or network isolation.

## Stream output and transfer files

```python
command = SandboxCommand(command="python -u check.py")
async for chunk in provider.stream(handle, command):
    print(chunk.sequence, chunk.stream, chunk.data)

await provider.write_file(handle, "/workspace/input.txt", b"review this")
download = await provider.read_file(handle, "/workspace/input.txt")
if download.ok:
    assert download.data == b"review this"
```

Output chunks have a monotonically increasing sequence within one stream. File and command results can include SHA-256 evidence digests. These digests show which bytes the adapter observed; they do not attest to the underlying isolation implementation.

## Enforce policy before every operation

```python
from anycode import PolicyEnforcer, PolicySandboxProvider

enforcer = PolicyEnforcer(policy_adapter, fail_closed=True)
secured_provider = PolicySandboxProvider(provider, enforcer)
```

The wrapper evaluates `create`, `execute`, `stream`, `file.write`, `file.read`, `cancel`, `snapshot`, and `destroy` separately. This prevents an allow decision for sandbox creation from becoming an unlimited capability for every later action.

## Connect a separate companion service

`CompanionSandboxAdapter(client, capabilities)` accepts a host-owned client that implements the sandbox methods. Declare the service's real isolation, networking, persistence, snapshot, streaming, cancellation, file, and evidence capabilities. Core AnyCode does not import or deploy the companion SDK.

Use this adapter when isolation must live in another trust zone, language, account, or cluster. Authenticate and authorize the client connection at the host boundary, and keep tenant separation and cleanup observable on both sides.

## Know the Daytona adapter limits

The current capability report declares remote isolation, allowlisted networking, persistent files, streaming, cancellation, file transfer, and evidence. It also declares these limitations:

- Isolation strength and placement depend on the selected Daytona runner and sandbox class.
- Cancellation force-stops the whole sandbox, not only one command.
- Stable point-in-time snapshots are not exposed by the adapter.

Check `provider.capabilities()` during startup. Refuse workloads whose requirements exceed the reported guarantees.

## Next steps

- [Propagate execution identity and policy](execution-identity.md)
- [Host AnyCode services](hosting-services.md)
- [Review the security boundary](../reference/security.md)
- [Deploy portable infrastructure](portable-infrastructure.md)
