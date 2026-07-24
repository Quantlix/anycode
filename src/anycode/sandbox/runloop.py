"""Runloop devbox sandbox provider adapter."""

from __future__ import annotations

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


class RunloopSandboxProvider:
    """Runloop devboxes; ``RUNLOOP_API_KEY`` stays in SDK configuration."""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    def _client_or_raise(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from runloop_api_client import AsyncRunloop  # type: ignore[import-not-found]
        except ImportError as error:
            message = (
                'RunloopSandboxProvider requires the optional "runloop_api_client" package. '
                'Install it with: pip install "anycode-py[sandbox-runloop]"'
            )
            raise ImportError(message) from error
        self._client = AsyncRunloop()
        return self._client

    async def create(self, spec: SandboxSpec) -> SandboxCreateResult:
        guard = first_guard_error(
            network_policy_error("runloop", spec, supported=("unrestricted",)),
            secrets_unsupported_error("runloop", spec),
        )
        if guard is not None:
            return SandboxCreateResult(ok=False, error=guard)
        try:
            devboxes = self._client_or_raise().devboxes
            kwargs: dict[str, Any] = {
                "metadata": {
                    **spec.labels,
                    "anycode.run_id": spec.run_id,
                    "anycode.tenant": spec.context.tenant_scope,
                    "anycode.classification": spec.context.classification,
                }
            }
            if spec.image:
                kwargs["blueprint_name"] = spec.image
            if spec.snapshot:
                kwargs["snapshot_id"] = spec.snapshot
            creator = getattr(devboxes, "create_and_await_running", None) or devboxes.create
            devbox = await call_maybe_async(creator, **kwargs)
            devbox_id = str(getattr(devbox, "id", ""))
            handle = SandboxHandle(
                id=devbox_id,
                provider="runloop",
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
            devboxes = self._client_or_raise().devboxes
            result = await call_maybe_async(devboxes.execute_sync, handle.id, command=shell_command(command))
            stdout = str(getattr(result, "stdout", "") or "")
            stderr = str(getattr(result, "stderr", "") or "")
            exit_code = int(getattr(result, "exit_status", None) or getattr(result, "exit_code", 0) or 0)
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
            devboxes = self._client_or_raise().devboxes
            writer = getattr(devboxes, "upload_file", None)
            if writer is not None:
                await call_maybe_async(writer, handle.id, path=path, file=data)
            else:
                await call_maybe_async(devboxes.write_file_contents, handle.id, file_path=path, contents=data.decode("utf-8", errors="replace"))
            return SandboxFileResult(ok=True, evidence=SandboxEvidence.from_bytes("file.write", data, metadata={"path": path}))
        except Exception as error:
            return SandboxFileResult(ok=False, error=ContractError(code="sandbox_file_write_failed", message=safe_exception_message(error)))

    async def read_file(self, handle: SandboxHandle, path: str) -> SandboxFileResult:
        try:
            devboxes = self._client_or_raise().devboxes
            reader = getattr(devboxes, "download_file", None) or devboxes.read_file_contents
            data = to_bytes(await call_maybe_async(reader, handle.id, file_path=path))
            return SandboxFileResult(ok=True, data=data, evidence=SandboxEvidence.from_bytes("file.read", data, metadata={"path": path}))
        except Exception as error:
            return SandboxFileResult(ok=False, error=ContractError(code="sandbox_file_read_failed", message=safe_exception_message(error)))

    async def cancel(self, handle: SandboxHandle) -> SandboxActionResult:
        try:
            devboxes = self._client_or_raise().devboxes
            await call_maybe_async(devboxes.suspend, handle.id)
            return SandboxActionResult(ok=True, evidence=SandboxEvidence.from_bytes("sandbox.cancel", handle.id.encode()))
        except Exception as error:
            return SandboxActionResult(ok=False, error=ContractError(code="sandbox_cancel_failed", message=safe_exception_message(error)))

    async def snapshot(self, handle: SandboxHandle, name: str) -> SandboxActionResult:
        try:
            devboxes = self._client_or_raise().devboxes
            snapshot = await call_maybe_async(devboxes.snapshot_disk, handle.id)
            reference = str(getattr(snapshot, "id", None) or snapshot)
            evidence = SandboxEvidence.from_bytes("sandbox.snapshot", reference.encode(), metadata={"name": name})
            return SandboxActionResult(ok=True, reference=reference, evidence=evidence)
        except Exception as error:
            return SandboxActionResult(ok=False, error=ContractError(code="sandbox_snapshot_failed", message=safe_exception_message(error)))

    async def destroy(self, handle: SandboxHandle) -> SandboxActionResult:
        try:
            devboxes = self._client_or_raise().devboxes
            await call_maybe_async(devboxes.shutdown, handle.id)
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
            provider="runloop",
            isolation="vm",
            networking="unrestricted",
            persistent_filesystem=True,
            snapshots=True,
            command_streaming=False,
            cancellation=True,
            file_transfer=True,
            evidence=True,
            limitations=(
                "Network egress cannot be restricted through the stable adapter; network='none' and allowlists fail closed.",
                "No managed secret store; sandbox secret references are rejected.",
                "Command streaming is buffered: output arrives after the command completes.",
                "Cancellation suspends the devbox; resume it from the Runloop console or API.",
            ),
        )
