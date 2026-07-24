"""Sandbox provider conformance, cleanup, evidence, and policy tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from anycode.contracts import PolicyDecision
from anycode.helpers.uuid7 import uuid7
from anycode.identity.context import ExecutionContext
from anycode.identity.policy import InMemoryPolicyAuditSink, PolicyEnforcer, PolicyRequest
from anycode.sandbox import (
    CompanionSandboxAdapter,
    DaytonaSandboxProvider,
    PolicySandboxProvider,
    SandboxCommand,
    SandboxProvider,
    SandboxSpec,
)


class FakeFileSystem:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    async def upload_file(self, data: bytes, path: str) -> None:
        self.files[path] = data

    async def download_file(self, path: str) -> bytes:
        return self.files[path]


class FakeProcess:
    def __init__(self) -> None:
        self.deleted_sessions: list[str] = []

    async def exec(self, command: str, **kwargs: object) -> object:
        del kwargs
        return SimpleNamespace(exit_code=0, result=f"ran:{command}", stderr="", artifacts=SimpleNamespace(stdout=f"ran:{command}"))

    async def create_session(self, session_id: str) -> None:
        del session_id

    async def execute_session_command(self, session_id: str, request: object, **kwargs: object) -> object:
        del session_id, request, kwargs
        return SimpleNamespace(cmd_id="command-1")

    async def get_session_command_logs_async(self, session_id: str, command_id: str, stdout: object, stderr: object) -> None:
        del session_id, command_id
        await stdout("hello")  # type: ignore[operator]
        await stderr("warning")  # type: ignore[operator]

    async def delete_session(self, session_id: str) -> None:
        self.deleted_sessions.append(session_id)


class FakeSandbox:
    def __init__(self) -> None:
        self.id = "sandbox-1"
        self.snapshot = None
        self.process = FakeProcess()
        self.fs = FakeFileSystem()
        self.stopped = False
        self.deleted = False

    async def stop(self, *, force: bool = False) -> None:
        self.stopped = force

    async def delete(self) -> None:
        self.deleted = True


class FakeDaytonaClient:
    def __init__(self) -> None:
        self.sandbox = FakeSandbox()

    async def create(self, params: object | None = None) -> FakeSandbox:
        del params
        return self.sandbox

    async def get(self, sandbox_id: str) -> FakeSandbox:
        assert sandbox_id == self.sandbox.id
        return self.sandbox


def _context() -> ExecutionContext:
    return ExecutionContext(principal="user:1", tenant_scope="tenant-a", classification="confidential", allowed_regions=("eu",))


def _spec() -> SandboxSpec:
    return SandboxSpec(run_id="run-1", task_id="task-1", correlation_id="corr-1", context=_context(), network="none")


async def test_daytona_provider_passes_lifecycle_command_file_stream_and_cleanup_contract() -> None:
    client = FakeDaytonaClient()
    provider = DaytonaSandboxProvider(client, session_request_factory=lambda **kwargs: SimpleNamespace(**kwargs))
    created = await provider.create(_spec())
    assert created.handle is not None
    handle = created.handle

    command = await provider.execute(handle, SandboxCommand(command="echo hello"))
    chunks = [chunk async for chunk in provider.stream(handle, SandboxCommand(command="long command"))]
    written = await provider.write_file(handle, "/workspace/file.txt", b"payload")
    read = await provider.read_file(handle, "/workspace/file.txt")
    canceled = await provider.cancel(handle)
    snapshot = await provider.snapshot(handle, "point-in-time")
    destroyed = await provider.destroy(handle)

    assert isinstance(provider, SandboxProvider)
    assert command.ok and command.stdout == "ran:echo hello" and command.evidence is not None
    assert [(chunk.stream, chunk.data) for chunk in chunks] == [("stdout", "hello"), ("stderr", "warning")]
    assert client.sandbox.process.deleted_sessions
    assert written.ok and read.ok and read.data == b"payload" and read.evidence is not None
    assert canceled.ok and client.sandbox.stopped
    assert not snapshot.ok and snapshot.error is not None and snapshot.error.code == "sandbox_snapshot_unsupported"
    assert destroyed.ok and client.sandbox.deleted
    assert provider.capabilities().command_streaming and provider.capabilities().cancellation
    assert not provider.capabilities().snapshots


async def test_companion_adapter_preserves_provider_contract() -> None:
    provider = DaytonaSandboxProvider(FakeDaytonaClient(), session_request_factory=lambda **kwargs: SimpleNamespace(**kwargs))
    adapter = CompanionSandboxAdapter(provider, provider.capabilities())
    created = await adapter.create(_spec())

    assert isinstance(adapter, SandboxProvider)
    assert created.ok and created.handle is not None
    assert adapter.capabilities() == provider.capabilities()


def test_sandbox_spec_rejects_network_and_plaintext_secret_misconfiguration() -> None:
    with pytest.raises(ValidationError):
        SandboxSpec(run_id="run", correlation_id="corr", context=_context(), network="none", allowed_domains=("example.com",))
    with pytest.raises(ValidationError):
        SandboxSpec(run_id="run", correlation_id="corr", context=_context(), secret_references={"API_KEY": "plaintext"})
    with pytest.raises(ValidationError):
        SandboxCommand(command="run", environment={"API_TOKEN": "plaintext"})


def test_sandbox_spec_accepts_any_provider_prefixed_secret_reference() -> None:
    spec = SandboxSpec(run_id="run", correlation_id="corr", context=_context(), secret_references={"API_KEY": "e2b:api-key"})
    assert spec.secret_references == {"API_KEY": "e2b:api-key"}


async def test_daytona_create_rejects_foreign_secret_prefix() -> None:
    provider = DaytonaSandboxProvider(FakeDaytonaClient(), session_request_factory=lambda **kwargs: SimpleNamespace(**kwargs))
    spec = SandboxSpec(run_id="run", correlation_id="corr", context=_context(), secret_references={"API_KEY": "modal:api-key"})

    created = await provider.create(spec)

    assert not created.ok and created.error is not None
    assert created.error.code == "sandbox_secret_reference_invalid"


class DenySandboxPolicy:
    async def decide(self, request: PolicyRequest) -> PolicyDecision:
        return PolicyDecision(
            id=str(uuid7()),
            run_id=request.run_id,
            task_id=request.task_id,
            outcome="deny",
            policy_version="policy/1",
            reason_codes=("sandbox_denied",),
            correlation_id=request.correlation_id,
            generation=request.generation,
            attempt=request.attempt,
        )


async def test_policy_wrapper_denies_before_sandbox_creation_and_audits_context() -> None:
    client = FakeDaytonaClient()
    base = DaytonaSandboxProvider(client, session_request_factory=lambda **kwargs: SimpleNamespace(**kwargs))
    audit = InMemoryPolicyAuditSink()
    provider = PolicySandboxProvider(base, PolicyEnforcer(DenySandboxPolicy(), fail_closed=True, audit_sink=audit))

    created = await provider.create(_spec())

    assert not created.ok and created.error is not None and created.error.code == "policy_denied"
    assert not client.sandbox.deleted and not client.sandbox.stopped
    assert len(audit.events) == 1 and audit.events[0].boundary == "sandbox"
    assert audit.events[0].context["tenant_scope"] == "tenant-a"
