"""Vercel Sandbox provider adapter."""

from __future__ import annotations

import sys
import types
from collections.abc import AsyncIterable
from typing import Any

from anycode.contracts.models import ContractError
from anycode.sandbox._base import (
    buffered_stream,
    call_maybe_async,
    first_guard_error,
    network_policy_error,
    read_streamish,
    secrets_unsupported_error,
    shell_command,
    snapshot_restore_unsupported_error,
    to_bytes,
)
from anycode.sandbox.models import (
    SandboxActionResult,
    SandboxCapabilities,
    SandboxCommand,
    SandboxCommandResult,
    SandboxCreateResult,
    SandboxEvidence,
    SandboxFileResult,
    SandboxHandle,
    SandboxHealth,
    SandboxOutputChunk,
    SandboxSpec,
)
from anycode.security.redaction import safe_exception_message


class VercelSandboxProvider:
    """Vercel Sandbox microVMs; auth stays in Vercel SDK env configuration."""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client
        self._sandboxes: dict[str, Any] = {}

    def _client_or_raise(self) -> Any:
        if self._client is not None:
            return self._client
        if sys.platform == "win32":
            # The Vercel SDK imports Unix-only pty modules (termios/tty) at
            # module scope for its interactive-shell helper, which this adapter
            # never calls; stub them so the sandbox API loads on Windows.
            for name in ("termios", "tty"):
                sys.modules.setdefault(name, types.ModuleType(name))
        try:
            from vercel.sandbox import AsyncSandbox  # type: ignore[import-not-found]
        except ImportError as error:
            message = 'VercelSandboxProvider requires the optional "vercel" package. Install it with: pip install "anycode-py[sandbox-vercel]"'
            raise ImportError(message) from error
        self._client = AsyncSandbox
        return self._client

    async def _resolve(self, handle: SandboxHandle) -> Any:
        sandbox = self._sandboxes.get(handle.id)
        if sandbox is not None:
            return sandbox
        sandbox = await call_maybe_async(self._client_or_raise().get, sandbox_id=handle.id)
        self._sandboxes[handle.id] = sandbox
        return sandbox

    async def create(self, spec: SandboxSpec) -> SandboxCreateResult:
        guard = first_guard_error(
            network_policy_error("vercel", spec, supported=("unrestricted",)),
            secrets_unsupported_error("vercel", spec),
            snapshot_restore_unsupported_error("vercel", spec),
        )
        if guard is not None:
            return SandboxCreateResult(ok=False, error=guard)
        try:
            sandbox_cls = self._client_or_raise()
            kwargs: dict[str, Any] = {}
            if spec.image:
                kwargs["runtime"] = spec.image
            sandbox = await call_maybe_async(sandbox_cls.create, **kwargs)
            sandbox_id = str(getattr(sandbox, "sandbox_id", None) or getattr(sandbox, "id", ""))
            self._sandboxes[sandbox_id] = sandbox
            handle = SandboxHandle(
                id=sandbox_id,
                provider="vercel",
                run_id=spec.run_id,
                task_id=spec.task_id,
                correlation_id=spec.correlation_id,
                context=spec.context,
                capabilities=self.capabilities(),
            )
            return SandboxCreateResult(ok=True, handle=handle)
        except Exception as error:
            return SandboxCreateResult(
                ok=False,
                error=ContractError(code="sandbox_create_failed", message=safe_exception_message(error), retryable=True),
            )

    async def execute(self, handle: SandboxHandle, command: SandboxCommand) -> SandboxCommandResult:
        try:
            sandbox = await self._resolve(handle)
            cmd = await call_maybe_async(sandbox.run_command, "bash", ["-lc", shell_command(command)])
            stdout = await read_streamish(getattr(cmd, "stdout", ""))
            stderr = await read_streamish(getattr(cmd, "stderr", ""))
            exit_code = int(getattr(cmd, "exit_code", 0) or 0)
            evidence = SandboxEvidence.from_bytes("command", f"{stdout}\0{stderr}".encode(), metadata={"exit_code": exit_code})
            return SandboxCommandResult(ok=exit_code == 0, exit_code=exit_code, stdout=stdout, stderr=stderr, evidence=evidence)
        except Exception as error:
            return SandboxCommandResult(
                ok=False,
                error=ContractError(code="sandbox_command_failed", message=safe_exception_message(error), retryable=True),
            )

    def stream(self, handle: SandboxHandle, command: SandboxCommand) -> AsyncIterable[SandboxOutputChunk]:
        return buffered_stream(self.execute, handle, command)

    async def write_file(self, handle: SandboxHandle, path: str, data: bytes) -> SandboxFileResult:
        try:
            sandbox = await self._resolve(handle)
            await call_maybe_async(sandbox.write_files, [{"path": path, "content": data}])
            return SandboxFileResult(ok=True, evidence=SandboxEvidence.from_bytes("file.write", data, metadata={"path": path}))
        except Exception as error:
            return SandboxFileResult(ok=False, error=ContractError(code="sandbox_file_write_failed", message=safe_exception_message(error)))

    async def read_file(self, handle: SandboxHandle, path: str) -> SandboxFileResult:
        try:
            sandbox = await self._resolve(handle)
            data = to_bytes(await call_maybe_async(sandbox.read_file, path))
            return SandboxFileResult(ok=True, data=data, evidence=SandboxEvidence.from_bytes("file.read", data, metadata={"path": path}))
        except Exception as error:
            return SandboxFileResult(ok=False, error=ContractError(code="sandbox_file_read_failed", message=safe_exception_message(error)))

    async def cancel(self, handle: SandboxHandle) -> SandboxActionResult:
        try:
            sandbox = await self._resolve(handle)
            await call_maybe_async(sandbox.stop)
            return SandboxActionResult(ok=True, evidence=SandboxEvidence.from_bytes("sandbox.cancel", handle.id.encode()))
        except Exception as error:
            return SandboxActionResult(ok=False, error=ContractError(code="sandbox_cancel_failed", message=safe_exception_message(error)))

    async def snapshot(self, handle: SandboxHandle, name: str) -> SandboxActionResult:
        del handle, name
        return SandboxActionResult(
            ok=False,
            error=ContractError(code="sandbox_snapshot_unsupported", message="Vercel Sandbox does not expose point-in-time snapshots."),
        )

    async def destroy(self, handle: SandboxHandle) -> SandboxActionResult:
        try:
            sandbox = await self._resolve(handle)
            await call_maybe_async(sandbox.stop)
            self._sandboxes.pop(handle.id, None)
            return SandboxActionResult(ok=True, evidence=SandboxEvidence.from_bytes("sandbox.destroy", handle.id.encode()))
        except Exception as error:
            return SandboxActionResult(ok=False, error=ContractError(code="sandbox_destroy_failed", message=safe_exception_message(error)))

    async def health(self) -> SandboxHealth:
        try:
            self._client_or_raise()
        except Exception as error:
            return SandboxHealth(status="unavailable", message=safe_exception_message(error))
        return SandboxHealth(status="healthy")

    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(
            provider="vercel",
            isolation="microvm",
            networking="unrestricted",
            persistent_filesystem=True,
            snapshots=False,
            command_streaming=False,
            cancellation=True,
            file_transfer=True,
            evidence=True,
            limitations=(
                "The Vercel Python SDK is in beta; pin the extra when reproducibility matters.",
                "spec.image selects a Vercel runtime string; arbitrary container images are unsupported.",
                "Network egress cannot be restricted through the stable adapter; network='none' and allowlists fail closed.",
                "No managed secret store; sandbox secret references are rejected.",
                "Command streaming is buffered: output arrives after the command completes.",
                "Sandbox lifetime is capped by the Vercel sandbox timeout.",
            ),
        )
