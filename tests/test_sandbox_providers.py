"""Factory and conformance tests for the expanded sandbox provider catalog."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from anycode.identity.context import ExecutionContext
from anycode.sandbox import (
    SANDBOX_PROVIDER_EXTRAS,
    DaytonaSandboxProvider,
    E2BSandboxProvider,
    SandboxCommand,
    SandboxProvider,
    SandboxSpec,
    create_sandbox_provider,
)


def _context() -> ExecutionContext:
    return ExecutionContext(principal="user:1", tenant_scope="tenant-a", classification="confidential", allowed_regions=("eu",))


def _spec(**overrides: object) -> SandboxSpec:
    defaults: dict = {
        "run_id": "run-1",
        "task_id": "task-1",
        "correlation_id": "corr-1",
        "context": _context(),
        "network": "unrestricted",
    }
    defaults.update(overrides)
    return SandboxSpec(**defaults)


async def _assert_lifecycle_conformance(provider: SandboxProvider, *, snapshots: bool) -> None:
    """Shared create -> execute -> stream -> file roundtrip -> cancel -> snapshot -> destroy contract."""
    assert isinstance(provider, SandboxProvider)
    created = await provider.create(_spec())
    assert created.ok and created.handle is not None
    handle = created.handle

    command = await provider.execute(handle, SandboxCommand(command="echo hello"))
    assert command.ok and command.exit_code == 0 and command.evidence is not None
    assert command.evidence.digest.startswith("sha256:")

    chunks = [chunk async for chunk in provider.stream(handle, SandboxCommand(command="echo hello"))]
    assert chunks and [chunk.sequence for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert all(chunk.stream in ("stdout", "stderr") for chunk in chunks)

    written = await provider.write_file(handle, "/workspace/file.txt", b"payload")
    read = await provider.read_file(handle, "/workspace/file.txt")
    assert written.ok and written.evidence is not None
    assert read.ok and read.data == b"payload"

    canceled = await provider.cancel(handle)
    assert canceled.ok

    snapshot = await provider.snapshot(handle, "point-in-time")
    if snapshots:
        assert snapshot.ok and snapshot.reference
    else:
        assert not snapshot.ok and snapshot.error is not None and snapshot.error.code == "sandbox_snapshot_unsupported"

    destroyed = await provider.destroy(handle)
    assert destroyed.ok
    assert (await provider.health()).status == "healthy"


# ---------------------------------------------------------------------------
# E2B fakes
# ---------------------------------------------------------------------------


class FakeE2BCommands:
    async def run(self, command: str, timeout: object = None) -> SimpleNamespace:
        del timeout
        return SimpleNamespace(stdout=f"ran:{command}", stderr="", exit_code=0)


class FakeE2BFiles:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    async def write(self, path: str, data: bytes) -> None:
        self.files[path] = data

    async def read(self, path: str) -> bytes:
        return self.files[path]


class FakeE2BSandbox:
    def __init__(self) -> None:
        self.sandbox_id = "e2b-1"
        self.commands = FakeE2BCommands()
        self.files = FakeE2BFiles()
        self.killed = False

    async def kill(self) -> None:
        self.killed = True


class FakeE2BSandboxClass:
    """Stands in for the e2b.AsyncSandbox class surface."""

    def __init__(self) -> None:
        self.sandbox = FakeE2BSandbox()

    async def create(self, **kwargs: object) -> FakeE2BSandbox:
        del kwargs
        return self.sandbox

    async def connect(self, sandbox_id: str) -> FakeE2BSandbox:
        assert sandbox_id == self.sandbox.sandbox_id
        return self.sandbox


async def test_e2b_provider_passes_lifecycle_conformance() -> None:
    provider = E2BSandboxProvider(FakeE2BSandboxClass())
    await _assert_lifecycle_conformance(provider, snapshots=False)


async def test_e2b_rejects_restricted_network_and_secrets() -> None:
    provider = E2BSandboxProvider(FakeE2BSandboxClass())

    denied_network = await provider.create(_spec(network="none"))
    assert not denied_network.ok and denied_network.error is not None
    assert denied_network.error.code == "sandbox_network_policy_unsupported"

    denied_secrets = await provider.create(_spec(secret_references={"API_KEY": "e2b:key"}))
    assert not denied_secrets.ok and denied_secrets.error is not None
    assert denied_secrets.error.code == "sandbox_secrets_unsupported"


async def test_e2b_reports_missing_sdk_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "e2b", None)
    provider = E2BSandboxProvider()

    created = await provider.create(_spec())

    assert not created.ok and created.error is not None
    assert "anycode-py[sandbox-e2b]" in created.error.message
    assert (await provider.health()).status == "unavailable"


def test_e2b_capabilities_are_honest() -> None:
    caps = E2BSandboxProvider(FakeE2BSandboxClass()).capabilities()
    assert caps.provider == "e2b" and caps.isolation == "microvm"
    assert not caps.snapshots and not caps.command_streaming


def test_factory_builds_daytona_provider() -> None:
    provider = create_sandbox_provider("daytona")
    assert isinstance(provider, DaytonaSandboxProvider)
    assert provider.capabilities().provider == "daytona"


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown sandbox provider: 'firecracker'"):
        create_sandbox_provider("firecracker")


def test_every_factory_provider_declares_an_extra() -> None:
    for name, extra in SANDBOX_PROVIDER_EXTRAS.items():
        provider = create_sandbox_provider(name)
        assert provider.capabilities().provider == name
        assert extra.startswith("sandbox")
