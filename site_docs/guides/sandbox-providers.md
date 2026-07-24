---
title: "Run AnyCode Work in Sandbox Providers"
description: Run AnyCode work in Daytona, E2B, Modal, Runloop, Vercel, or LangSmith sandboxes behind one protocol with network policy, secret references, and evidence.
keywords: AnyCode sandbox, Daytona E2B Modal Runloop Vercel LangSmith sandbox, AI agent code execution, policy sandbox provider, isolated agent tools
---

# Run work through sandbox providers

The `SandboxProvider` protocol gives AnyCode one explicit boundary for remote execution, files, streaming, cancellation, cleanup, health, and evidence. Six remote backends implement it — Daytona, E2B, Modal, Runloop, Vercel Sandbox, and LangSmith — plus `CompanionSandboxAdapter` for a separately deployed sandbox service. Wrap any provider with `PolicySandboxProvider` when every operation needs an external authorization decision.

## Choose a provider

Each backend ships behind its own install extra, and `create_sandbox_provider(name)` builds any of them without importing the SDK until first use:

```python
from anycode import create_sandbox_provider

provider = create_sandbox_provider("e2b")  # or daytona, modal, runloop, vercel, langsmith
print(provider.capabilities())
```

| Provider | Extra | Isolation | Snapshots | Live streaming | Network policy | Secret references |
| --- | --- | --- | --- | --- | --- | --- |
| `daytona` | `sandbox` | remote | no | yes | none / allowlist | `daytona:<name>` |
| `e2b` | `sandbox-e2b` | microVM | no | buffered | unrestricted only | rejected |
| `modal` | `sandbox-modal` | container (gVisor) | yes | buffered | none / CIDR allowlist | `modal:<name>` |
| `runloop` | `sandbox-runloop` | VM | yes | buffered | unrestricted only | rejected |
| `vercel` | `sandbox-vercel` | microVM | no | buffered | unrestricted only | rejected |
| `langsmith` | `sandbox-langsmith` | microVM | no | buffered | unrestricted only | rejected |

Every backend fails closed: when a spec requests a network mode, snapshot restore, or secret scheme the provider cannot enforce, `create()` returns a typed error (`sandbox_network_policy_unsupported`, `sandbox_snapshot_unsupported`, `sandbox_secrets_unsupported`, or `sandbox_secret_reference_invalid`) instead of silently granting more than requested. "Buffered" streaming satisfies the streaming contract but delivers output after the command completes. Check `provider.capabilities()` — including its `limitations` tuple — before trusting a workload to a backend.

Credentials always stay in each SDK's own configuration (`E2B_API_KEY`, Modal tokens, `RUNLOOP_API_KEY`, `VERCEL_TOKEN`/`VERCEL_OIDC_TOKEN` with team and project ids, `LANGSMITH_API_KEY`). Do not place credentials in `SandboxSpec`, a command environment, labels, logs, or durable run data.

## Install the Daytona integration

```bash
uv add "anycode-py[sandbox]"
```

Configure Daytona credentials through the SDK's supported host configuration.

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
| `secret_references` | Every value is a provider-prefixed reference (e.g. `daytona:token-name`); each backend accepts only its own prefix |
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

## The complete, runnable program

The snippets above assume a live Daytona host. Here is a complete file you can run offline with no credentials and no optional dependencies. It implements a small in-memory sandbox behind `CompanionSandboxAdapter` — exactly the companion-service shape described above — wraps it with `PolicySandboxProvider` so every operation needs an allow decision, and exercises the full create, execute, stream, file, and destroy path. Swap `InMemorySandboxClient` for `DaytonaSandboxProvider()` (or your own companion client) to run against a real isolated host; the calling code does not change.

```python title="sandbox_providers.py"
import asyncio
from collections.abc import AsyncIterator

from anycode import (
    CompanionSandboxAdapter,
    ExecutionContext,
    PolicyDecision,
    PolicyEnforcer,
    PolicyRequest,
    PolicySandboxProvider,
    SandboxCommand,
    SandboxSpec,
    uuid7,
)
from anycode.sandbox import (
    SandboxActionResult,
    SandboxCapabilities,
    SandboxCommandResult,
    SandboxCreateResult,
    SandboxEvidence,
    SandboxFileResult,
    SandboxHandle,
    SandboxHealth,
    SandboxOutputChunk,
)


def build_capabilities(provider: str) -> SandboxCapabilities:
    return SandboxCapabilities(
        provider=provider,
        isolation="remote",
        networking="allowlist",
        persistent_filesystem=True,
        snapshots=False,
        command_streaming=True,
        cancellation=True,
        file_transfer=True,
        evidence=True,
        limitations=("Stable point-in-time snapshots are not exposed.",),
    )


class InMemorySandboxClient:
    """Offline stand-in for a separately deployed sandbox companion service."""

    def __init__(self, provider: str) -> None:
        self._provider = provider
        self._capabilities = build_capabilities(provider)
        self._files: dict[str, bytes] = {}

    async def create(self, spec: SandboxSpec) -> SandboxCreateResult:
        handle = SandboxHandle(
            id="sandbox-1",
            provider=self._provider,
            run_id=spec.run_id,
            task_id=spec.task_id,
            correlation_id=spec.correlation_id,
            context=spec.context,
            capabilities=self._capabilities,
        )
        return SandboxCreateResult(ok=True, handle=handle)

    async def execute(self, handle: SandboxHandle, command: SandboxCommand) -> SandboxCommandResult:
        stdout = f"ran: {command.command}\n"
        return SandboxCommandResult(
            ok=True,
            exit_code=0,
            stdout=stdout,
            evidence=SandboxEvidence.from_bytes("execute", stdout.encode()),
        )

    async def stream(self, handle: SandboxHandle, command: SandboxCommand) -> AsyncIterator[SandboxOutputChunk]:
        yield SandboxOutputChunk(stream="stdout", data=f"starting: {command.command}", sequence=1)
        yield SandboxOutputChunk(stream="stdout", data="done", sequence=2)

    async def write_file(self, handle: SandboxHandle, path: str, data: bytes) -> SandboxFileResult:
        self._files[path] = data
        return SandboxFileResult(ok=True, evidence=SandboxEvidence.from_bytes("file.write", data))

    async def read_file(self, handle: SandboxHandle, path: str) -> SandboxFileResult:
        data = self._files[path]
        return SandboxFileResult(ok=True, data=data, evidence=SandboxEvidence.from_bytes("file.read", data))

    async def cancel(self, handle: SandboxHandle) -> SandboxActionResult:
        return SandboxActionResult(ok=True)

    async def snapshot(self, handle: SandboxHandle, name: str) -> SandboxActionResult:
        return SandboxActionResult(ok=True)

    async def destroy(self, handle: SandboxHandle) -> SandboxActionResult:
        return SandboxActionResult(ok=True)

    async def health(self) -> SandboxHealth:
        return SandboxHealth(status="healthy")


class AllowSandboxPolicy:
    """Minimal external policy adapter; a real deployment calls its policy service."""

    async def decide(self, request: PolicyRequest) -> PolicyDecision:
        return PolicyDecision(
            id=str(uuid7()),
            run_id=request.run_id,
            task_id=request.task_id,
            outcome="allow",
            policy_version="sandbox-policy/1",
            reason_codes=("sandbox_allowed",),
            correlation_id=request.correlation_id,
            causation_id=request.causation_id,
            generation=request.generation,
            attempt=request.attempt,
        )


async def main() -> None:
    provider_name = "companion-demo"
    base = CompanionSandboxAdapter(InMemorySandboxClient(provider_name), build_capabilities(provider_name))
    secured = PolicySandboxProvider(base, PolicyEnforcer(AllowSandboxPolicy(), fail_closed=True))

    caps = secured.capabilities()
    print(f"provider={caps.provider} isolation={caps.isolation} snapshots={caps.snapshots}")

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
        labels={"workload": "dependency-check"},
    )

    created = await secured.create(spec)
    if not created.ok or created.handle is None:
        raise RuntimeError(created.error.message if created.error else "create failed")
    handle = created.handle

    try:
        result = await secured.execute(handle, SandboxCommand(command="python --version", timeout_seconds=30))
        if not result.ok:
            raise RuntimeError(result.error.message if result.error else result.stderr)
        print("stdout:", result.stdout.strip())
        print("evidence:", result.evidence.digest if result.evidence else "none")

        async for chunk in secured.stream(handle, SandboxCommand(command="python -u check.py")):
            print(f"stream[{chunk.sequence}] {chunk.stream}: {chunk.data}")

        await secured.write_file(handle, "/workspace/input.txt", b"review this")
        download = await secured.read_file(handle, "/workspace/input.txt")
        if download.ok and download.data is not None:
            print("read back:", download.data.decode())
    finally:
        await secured.destroy(handle)


if __name__ == "__main__":
    asyncio.run(main())
```

Run it from the project root:

```bash
uv run python sandbox_providers.py
```

!!! tip "Tested copy"
    The real `DaytonaSandboxProvider` and `PolicySandboxProvider` behavior — lifecycle, streaming, file transfer, evidence, and the policy-denied path — is exercised in [`tests/test_sandbox.py`](https://github.com/Quantlix/anycode/blob/main/tests/test_sandbox.py). See also [`examples/40_operational_portability.py`](https://github.com/Quantlix/anycode/blob/main/examples/40_operational_portability.py) for the shared identity and policy envelope.

## Next steps

- [Propagate execution identity and policy](execution-identity.md)
- [Host AnyCode services](hosting-services.md)
- [Review the security boundary](../reference/security.md)
- [Deploy portable infrastructure](portable-infrastructure.md)
