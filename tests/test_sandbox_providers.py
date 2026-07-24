"""Factory and conformance tests for the expanded sandbox provider catalog."""

from __future__ import annotations

import base64
import re
import sys
from types import SimpleNamespace

import pytest

from anycode.identity.context import ExecutionContext
from anycode.sandbox import (
    SANDBOX_PROVIDER_EXTRAS,
    DaytonaSandboxProvider,
    E2BSandboxProvider,
    LangSmithSandboxProvider,
    ModalSandboxProvider,
    RunloopSandboxProvider,
    SandboxCommand,
    SandboxProvider,
    SandboxSpec,
    VercelSandboxProvider,
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


# ---------------------------------------------------------------------------
# Modal fakes
# ---------------------------------------------------------------------------


class FakeModalFile:
    def __init__(self, store: dict[str, bytes], path: str, mode: str) -> None:
        self._store = store
        self._path = path
        self._mode = mode

    async def write(self, data: bytes) -> None:
        self._store[self._path] = data

    async def read(self) -> bytes:
        return self._store[self._path]

    async def close(self) -> None:
        return None


class FakeModalProcess:
    def __init__(self, command: str) -> None:
        self.stdout = f"ran:{command}"
        self.stderr = ""
        self.returncode = 0

    async def wait(self) -> None:
        return None


class FakeModalFilesystem:
    """Mirrors modal's Sandbox.filesystem API (write_bytes takes data first)."""

    def __init__(self, store: dict[str, bytes]) -> None:
        self._store = store

    async def write_bytes(self, data: bytes, remote_path: str) -> None:
        self._store[remote_path] = data

    async def read_bytes(self, remote_path: str) -> bytes:
        return self._store[remote_path]


class FakeModalSandbox:
    def __init__(self) -> None:
        self.object_id = "modal-1"
        self.files: dict[str, bytes] = {}
        self.filesystem = FakeModalFilesystem(self.files)
        self.terminated = False
        self.create_kwargs: dict[str, object] = {}

    async def exec(self, *args: str, **kwargs: object) -> FakeModalProcess:
        del kwargs
        return FakeModalProcess(args[-1])

    async def open(self, path: str, mode: str) -> FakeModalFile:
        return FakeModalFile(self.files, path, mode)

    async def terminate(self) -> None:
        self.terminated = True

    async def snapshot_filesystem(self) -> SimpleNamespace:
        return SimpleNamespace(object_id="im-snapshot-1")


class FakeModalModule:
    """Stands in for the imported modal module surface."""

    def __init__(self) -> None:
        self.sandbox = FakeModalSandbox()
        self.secret_names: list[str] = []
        outer = self

        class App:
            @staticmethod
            async def lookup(name: str, *, create_if_missing: bool = False) -> SimpleNamespace:
                del create_if_missing
                return SimpleNamespace(name=name)

        class Image:
            @staticmethod
            def from_registry(image: str) -> SimpleNamespace:
                return SimpleNamespace(image=image)

        class Secret:
            @staticmethod
            def from_name(name: str) -> SimpleNamespace:
                outer.secret_names.append(name)
                return SimpleNamespace(name=name)

        class Sandbox:
            @staticmethod
            async def create(**kwargs: object) -> FakeModalSandbox:
                outer.sandbox.create_kwargs = dict(kwargs)
                return outer.sandbox

            @staticmethod
            async def from_id(sandbox_id: str) -> FakeModalSandbox:
                assert sandbox_id == outer.sandbox.object_id
                return outer.sandbox

        self.App = App
        self.Image = Image
        self.Secret = Secret
        self.Sandbox = Sandbox


async def test_modal_provider_passes_lifecycle_conformance() -> None:
    provider = ModalSandboxProvider(FakeModalModule())
    await _assert_lifecycle_conformance(provider, snapshots=True)


async def test_modal_maps_network_and_secrets_to_sdk_arguments() -> None:
    modal_mod = FakeModalModule()
    provider = ModalSandboxProvider(modal_mod)

    created = await provider.create(_spec(network="none", secret_references={"API_KEY": "modal:prod-key"}))

    assert created.ok
    assert modal_mod.sandbox.create_kwargs["block_network"] is True
    assert modal_mod.secret_names == ["prod-key"]


async def test_modal_fails_closed_on_domain_allowlists_and_foreign_secrets() -> None:
    provider = ModalSandboxProvider(FakeModalModule())

    domains = await provider.create(_spec(network="allowlist", allowed_domains=("example.com",)))
    assert not domains.ok and domains.error is not None
    assert domains.error.code == "sandbox_network_policy_unsupported"

    foreign = await provider.create(_spec(secret_references={"API_KEY": "daytona:key"}))
    assert not foreign.ok and foreign.error is not None
    assert foreign.error.code == "sandbox_secret_reference_invalid"


async def test_modal_reports_missing_sdk_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "modal", None)
    provider = ModalSandboxProvider()

    created = await provider.create(_spec())

    assert not created.ok and created.error is not None
    assert "anycode-py[sandbox-modal]" in created.error.message


# ---------------------------------------------------------------------------
# Runloop fakes
# ---------------------------------------------------------------------------


class FakeRunloopDevboxes:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.suspended = False
        self.shutdown_called = False

    async def create(self, **kwargs: object) -> SimpleNamespace:
        del kwargs
        return SimpleNamespace(id="devbox-1")

    async def execute_sync(self, devbox_id: str, *, command: str) -> SimpleNamespace:
        assert devbox_id == "devbox-1"
        return SimpleNamespace(stdout=f"ran:{command}", stderr="", exit_status=0)

    async def write_file_contents(self, devbox_id: str, *, file_path: str, contents: str) -> None:
        assert devbox_id == "devbox-1"
        self.files[file_path] = contents.encode("utf-8")

    async def read_file_contents(self, devbox_id: str, *, file_path: str) -> str:
        assert devbox_id == "devbox-1"
        return self.files[file_path].decode("utf-8")

    async def suspend(self, devbox_id: str) -> None:
        assert devbox_id == "devbox-1"
        self.suspended = True

    async def snapshot_disk(self, devbox_id: str) -> SimpleNamespace:
        assert devbox_id == "devbox-1"
        return SimpleNamespace(id="snapshot-1")

    async def shutdown(self, devbox_id: str) -> None:
        assert devbox_id == "devbox-1"
        self.shutdown_called = True


class FakeRunloopClient:
    def __init__(self) -> None:
        self.devboxes = FakeRunloopDevboxes()


async def test_runloop_provider_passes_lifecycle_conformance() -> None:
    provider = RunloopSandboxProvider(FakeRunloopClient())
    await _assert_lifecycle_conformance(provider, snapshots=True)


async def test_runloop_maps_image_and_snapshot_to_blueprint_arguments() -> None:
    client = FakeRunloopClient()
    captured: dict[str, object] = {}

    async def create(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(id="devbox-1")

    client.devboxes.create = create  # type: ignore[method-assign]
    provider = RunloopSandboxProvider(client)

    assert (await provider.create(_spec(image="py-blueprint"))).ok
    assert captured["blueprint_name"] == "py-blueprint"

    assert (await provider.create(_spec(snapshot="snap-9"))).ok
    assert captured["snapshot_id"] == "snap-9"


async def test_runloop_rejects_restricted_network_and_secrets() -> None:
    provider = RunloopSandboxProvider(FakeRunloopClient())

    denied_network = await provider.create(_spec(network="none"))
    assert not denied_network.ok and denied_network.error is not None
    assert denied_network.error.code == "sandbox_network_policy_unsupported"

    denied_secrets = await provider.create(_spec(secret_references={"API_KEY": "runloop:key"}))
    assert not denied_secrets.ok and denied_secrets.error is not None
    assert denied_secrets.error.code == "sandbox_secrets_unsupported"


async def test_runloop_reports_missing_sdk_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "runloop_api_client", None)
    provider = RunloopSandboxProvider()

    created = await provider.create(_spec())

    assert not created.ok and created.error is not None
    assert "anycode-py[sandbox-runloop]" in created.error.message


# ---------------------------------------------------------------------------
# Vercel fakes
# ---------------------------------------------------------------------------


class FakeVercelCommand:
    def __init__(self, command: str) -> None:
        self.exit_code = 0
        self._stdout = f"ran:{command}"

    async def stdout(self) -> str:
        return self._stdout

    async def stderr(self) -> str:
        return ""


class FakeVercelSandbox:
    def __init__(self) -> None:
        self.sandbox_id = "vercel-1"
        self.files: dict[str, bytes] = {}
        self.stopped = False
        self.create_kwargs: dict[str, object] = {}

    async def run_command(self, command: str, args: list[str]) -> FakeVercelCommand:
        del command
        return FakeVercelCommand(args[-1])

    async def write_files(self, files: list[dict[str, object]]) -> None:
        for entry in files:
            self.files[str(entry["path"])] = bytes(entry["content"])  # type: ignore[arg-type]

    async def read_file(self, path: str) -> bytes:
        return self.files[path]

    async def stop(self) -> None:
        self.stopped = True


class FakeVercelSandboxClass:
    """Stands in for the vercel.sandbox.AsyncSandbox class surface."""

    def __init__(self) -> None:
        self.sandbox = FakeVercelSandbox()

    async def create(self, **kwargs: object) -> FakeVercelSandbox:
        self.sandbox.create_kwargs = dict(kwargs)
        return self.sandbox

    async def get(self, *, sandbox_id: str) -> FakeVercelSandbox:
        assert sandbox_id == self.sandbox.sandbox_id
        return self.sandbox


async def test_vercel_provider_passes_lifecycle_conformance() -> None:
    provider = VercelSandboxProvider(FakeVercelSandboxClass())
    await _assert_lifecycle_conformance(provider, snapshots=False)


async def test_vercel_maps_image_to_runtime_and_rejects_restrictions() -> None:
    sandbox_cls = FakeVercelSandboxClass()
    provider = VercelSandboxProvider(sandbox_cls)

    created = await provider.create(_spec(image="python3.13"))
    assert created.ok and sandbox_cls.sandbox.create_kwargs == {"runtime": "python3.13"}

    denied_network = await provider.create(_spec(network="allowlist", allowed_cidrs=("10.0.0.0/8",)))
    assert not denied_network.ok and denied_network.error is not None
    assert denied_network.error.code == "sandbox_network_policy_unsupported"

    denied_secrets = await provider.create(_spec(secret_references={"API_KEY": "vercel:key"}))
    assert not denied_secrets.ok and denied_secrets.error is not None
    assert denied_secrets.error.code == "sandbox_secrets_unsupported"


async def test_vercel_reports_missing_sdk_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "vercel", None)
    monkeypatch.setitem(sys.modules, "vercel.sandbox", None)
    provider = VercelSandboxProvider()

    created = await provider.create(_spec())

    assert not created.ok and created.error is not None
    assert "anycode-py[sandbox-vercel]" in created.error.message


# ---------------------------------------------------------------------------
# LangSmith fakes
# ---------------------------------------------------------------------------


class FakeLangSmithSandbox:
    """run()-only sandbox with a tiny in-memory filesystem for the base64 shell transfers."""

    def __init__(self) -> None:
        self.id = "ls-1"
        self.files: dict[str, bytes] = {}
        self.deleted = False

    async def run(self, command: str) -> SimpleNamespace:
        write_match = re.search(r"printf %s (\S+) \| base64 -d > (.+)$", command)
        if write_match:
            encoded = write_match.group(1).strip("'\"")
            path = write_match.group(2).strip("'\"")
            self.files[path] = base64.b64decode(encoded)
            return SimpleNamespace(stdout="", stderr="", exit_code=0)
        read_match = re.match(r"^base64 (.+)$", command)
        if read_match:
            path = read_match.group(1).strip("'\"")
            if path not in self.files:
                return SimpleNamespace(stdout="", stderr=f"base64: {path}: No such file", exit_code=1)
            return SimpleNamespace(stdout=base64.b64encode(self.files[path]).decode("ascii"), stderr="", exit_code=0)
        return SimpleNamespace(stdout=f"ran:{command}", stderr="", exit_code=0)

    async def delete(self) -> None:
        self.deleted = True


class FakeLangSmithClient:
    def __init__(self) -> None:
        self.sandbox = FakeLangSmithSandbox()

    async def create_sandbox(self, **kwargs: object) -> FakeLangSmithSandbox:
        del kwargs
        return self.sandbox


async def test_langsmith_provider_passes_lifecycle_conformance() -> None:
    provider = LangSmithSandboxProvider(FakeLangSmithClient())
    await _assert_lifecycle_conformance(provider, snapshots=False)


async def test_langsmith_read_missing_file_fails_with_shell_error() -> None:
    provider = LangSmithSandboxProvider(FakeLangSmithClient())
    created = await provider.create(_spec())
    assert created.handle is not None

    read = await provider.read_file(created.handle, "/workspace/missing.txt")

    assert not read.ok and read.error is not None and read.error.code == "sandbox_file_read_failed"


async def test_langsmith_rejects_restricted_network_and_secrets() -> None:
    provider = LangSmithSandboxProvider(FakeLangSmithClient())

    denied_network = await provider.create(_spec(network="none"))
    assert not denied_network.ok and denied_network.error is not None
    assert denied_network.error.code == "sandbox_network_policy_unsupported"

    denied_secrets = await provider.create(_spec(secret_references={"API_KEY": "langsmith:key"}))
    assert not denied_secrets.ok and denied_secrets.error is not None
    assert denied_secrets.error.code == "sandbox_secrets_unsupported"


async def test_langsmith_reports_missing_sdk_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "langsmith", None)
    monkeypatch.setitem(sys.modules, "langsmith.sandbox", None)
    provider = LangSmithSandboxProvider()

    created = await provider.create(_spec())

    assert not created.ok and created.error is not None
    assert "anycode-py[sandbox-langsmith]" in created.error.message


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
