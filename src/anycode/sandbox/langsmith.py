"""LangSmith sandbox provider adapter."""

from __future__ import annotations

import base64
import shlex
from collections.abc import AsyncIterable
from typing import Any

from anycode.contracts.models import ContractError
from anycode.sandbox._base import (
    buffered_stream,
    call_maybe_async,
    first_guard_error,
    network_policy_error,
    secrets_unsupported_error,
    shell_command,
    snapshot_restore_unsupported_error,
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


class LangSmithSandboxProvider:
    """LangSmith microVM sandboxes; ``LANGSMITH_API_KEY`` stays in SDK configuration."""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client
        self._sandboxes: dict[str, Any] = {}

    def _client_or_raise(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from langsmith import sandbox as ls_sandbox  # type: ignore[import-not-found]
        except ImportError as error:
            message = (
                'LangSmithSandboxProvider requires the optional "langsmith[sandbox]" package. '
                'Install it with: pip install "anycode-py[sandbox-langsmith]"'
            )
            raise ImportError(message) from error
        client_cls = getattr(ls_sandbox, "AsyncSandboxClient", None) or ls_sandbox.SandboxClient
        self._client = client_cls()
        return self._client

    async def _resolve(self, handle: SandboxHandle) -> Any:
        sandbox = self._sandboxes.get(handle.id)
        if sandbox is not None:
            return sandbox
        client = self._client_or_raise()
        getter = getattr(client, "get_sandbox", None)
        if getter is None:
            raise LookupError(f"Sandbox {handle.id!r} is not tracked by this provider instance")
        sandbox = await call_maybe_async(getter, handle.id)
        self._sandboxes[handle.id] = sandbox
        return sandbox

    async def create(self, spec: SandboxSpec) -> SandboxCreateResult:
        guard = first_guard_error(
            network_policy_error("langsmith", spec, supported=("unrestricted",)),
            secrets_unsupported_error("langsmith", spec),
            snapshot_restore_unsupported_error("langsmith", spec),
        )
        if guard is not None:
            return SandboxCreateResult(ok=False, error=guard)
        try:
            client = self._client_or_raise()
            creator = getattr(client, "create_sandbox", None) or client.create
            kwargs: dict[str, Any] = {}
            if spec.image:
                kwargs["image"] = spec.image
            sandbox = await call_maybe_async(creator, **kwargs)
            sandbox_id = str(getattr(sandbox, "id", None) or getattr(sandbox, "sandbox_id", None) or getattr(sandbox, "name", ""))
            self._sandboxes[sandbox_id] = sandbox
            handle = SandboxHandle(
                id=sandbox_id,
                provider="langsmith",
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

    async def _run(self, handle: SandboxHandle, command_text: str) -> tuple[str, str, int]:
        sandbox = await self._resolve(handle)
        result = await call_maybe_async(sandbox.run, command_text)
        stdout = str(getattr(result, "stdout", "") or "")
        stderr = str(getattr(result, "stderr", "") or "")
        exit_code = int(getattr(result, "exit_code", None) or getattr(result, "returncode", 0) or 0)
        return stdout, stderr, exit_code

    async def execute(self, handle: SandboxHandle, command: SandboxCommand) -> SandboxCommandResult:
        try:
            stdout, stderr, exit_code = await self._run(handle, shell_command(command))
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
            encoded = base64.b64encode(data).decode("ascii")
            quoted = shlex.quote(path)
            stdout, stderr, exit_code = await self._run(
                handle, f'mkdir -p "$(dirname {quoted})" && printf %s {shlex.quote(encoded)} | base64 -d > {quoted}'
            )
            if exit_code != 0:
                return SandboxFileResult(ok=False, error=ContractError(code="sandbox_file_write_failed", message=stderr or stdout))
            return SandboxFileResult(ok=True, evidence=SandboxEvidence.from_bytes("file.write", data, metadata={"path": path}))
        except Exception as error:
            return SandboxFileResult(ok=False, error=ContractError(code="sandbox_file_write_failed", message=safe_exception_message(error)))

    async def read_file(self, handle: SandboxHandle, path: str) -> SandboxFileResult:
        try:
            stdout, stderr, exit_code = await self._run(handle, f"base64 {shlex.quote(path)}")
            if exit_code != 0:
                return SandboxFileResult(ok=False, error=ContractError(code="sandbox_file_read_failed", message=stderr or stdout))
            data = base64.b64decode("".join(stdout.split()))
            return SandboxFileResult(ok=True, data=data, evidence=SandboxEvidence.from_bytes("file.read", data, metadata={"path": path}))
        except Exception as error:
            return SandboxFileResult(ok=False, error=ContractError(code="sandbox_file_read_failed", message=safe_exception_message(error)))

    async def _delete(self, handle: SandboxHandle) -> None:
        sandbox = self._sandboxes.get(handle.id)
        deleter = getattr(sandbox, "delete", None) if sandbox is not None else None
        if deleter is not None:
            await call_maybe_async(deleter)
            return
        client = self._client_or_raise()
        await call_maybe_async(getattr(client, "delete_sandbox"), handle.id)

    async def cancel(self, handle: SandboxHandle) -> SandboxActionResult:
        try:
            await self._delete(handle)
            return SandboxActionResult(ok=True, evidence=SandboxEvidence.from_bytes("sandbox.cancel", handle.id.encode()))
        except Exception as error:
            return SandboxActionResult(ok=False, error=ContractError(code="sandbox_cancel_failed", message=safe_exception_message(error)))

    async def snapshot(self, handle: SandboxHandle, name: str) -> SandboxActionResult:
        del handle, name
        return SandboxActionResult(
            ok=False,
            error=ContractError(code="sandbox_snapshot_unsupported", message="LangSmith sandboxes do not expose point-in-time snapshots."),
        )

    async def destroy(self, handle: SandboxHandle) -> SandboxActionResult:
        try:
            await self._delete(handle)
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
            provider="langsmith",
            isolation="microvm",
            networking="unrestricted",
            persistent_filesystem=True,
            snapshots=False,
            command_streaming=False,
            cancellation=True,
            file_transfer=True,
            evidence=True,
            limitations=(
                "Network egress cannot be restricted through the stable adapter; network='none' and allowlists fail closed.",
                "No managed secret store; sandbox secret references are rejected.",
                "Command streaming is buffered: output arrives after the command completes.",
                "File transfer runs through in-sandbox shell base64, so the image must provide a POSIX shell and base64.",
                "Cancellation deletes the sandbox rather than only one command.",
            ),
        )
