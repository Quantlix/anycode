"""Modal sandbox provider adapter."""

from __future__ import annotations

from collections.abc import AsyncIterable
from typing import Any

from anycode.contracts.models import ContractError
from anycode.sandbox._base import (
    buffered_stream,
    call_maybe_async,
    first_guard_error,
    read_streamish,
    shell_command,
    snapshot_restore_unsupported_error,
    to_bytes,
)
from anycode.sandbox._secrets import strip_secret_prefix, validate_secret_references
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


class ModalSandboxProvider:
    """Modal gVisor sandboxes; tokens stay in Modal's own configuration."""

    def __init__(self, client: Any | None = None, *, app_name: str = "anycode-sandbox") -> None:
        self._client = client
        self._app_name = app_name
        self._sandboxes: dict[str, Any] = {}

    def _client_or_raise(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import modal  # type: ignore[import-not-found]
        except ImportError as error:
            message = 'ModalSandboxProvider requires the optional "modal" package. Install it with: pip install "anycode-py[sandbox-modal]"'
            raise ImportError(message) from error
        self._client = modal
        return self._client

    def _network_guard(self, spec: SandboxSpec) -> ContractError | None:
        if spec.network == "allowlist" and spec.allowed_domains:
            return ContractError(
                code="sandbox_network_policy_unsupported",
                message="modal enforces CIDR allowlists only; domain allowlists fail closed",
            )
        return None

    async def _resolve(self, handle: SandboxHandle) -> Any:
        sandbox = self._sandboxes.get(handle.id)
        if sandbox is not None:
            return sandbox
        sandbox = await call_maybe_async(self._client_or_raise().Sandbox.from_id, handle.id)
        self._sandboxes[handle.id] = sandbox
        return sandbox

    async def create(self, spec: SandboxSpec) -> SandboxCreateResult:
        guard = first_guard_error(
            self._network_guard(spec),
            validate_secret_references("modal", spec),
            snapshot_restore_unsupported_error("modal", spec),
        )
        if guard is not None:
            return SandboxCreateResult(ok=False, error=guard)
        try:
            modal_mod = self._client_or_raise()
            app = await call_maybe_async(modal_mod.App.lookup, self._app_name, create_if_missing=True)
            image = await call_maybe_async(modal_mod.Image.from_registry, spec.image) if spec.image else None
            kwargs: dict[str, Any] = {"app": app}
            if image is not None:
                kwargs["image"] = image
            if spec.network == "none":
                kwargs["block_network"] = True
            elif spec.network == "allowlist":
                kwargs["cidr_allowlist"] = list(spec.allowed_cidrs)
            if spec.secret_references:
                kwargs["secrets"] = [
                    await call_maybe_async(modal_mod.Secret.from_name, strip_secret_prefix("modal", reference))
                    for reference in spec.secret_references.values()
                ]
            sandbox = await call_maybe_async(modal_mod.Sandbox.create, **kwargs)
            sandbox_id = str(getattr(sandbox, "object_id", None) or getattr(sandbox, "id", ""))
            self._sandboxes[sandbox_id] = sandbox
            handle = SandboxHandle(
                id=sandbox_id,
                provider="modal",
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
            kwargs: dict[str, Any] = {}
            if command.timeout_seconds is not None:
                kwargs["timeout"] = command.timeout_seconds
            process = await call_maybe_async(sandbox.exec, "bash", "-lc", shell_command(command), **kwargs)
            waiter = getattr(process, "wait", None)
            if callable(waiter):
                await call_maybe_async(waiter)
            stdout = await read_streamish(getattr(process, "stdout", ""))
            stderr = await read_streamish(getattr(process, "stderr", ""))
            exit_code = int(getattr(process, "returncode", 0) or 0)
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
            # Modal deprecated Sandbox.open() in favor of Sandbox.filesystem;
            # keep the open() path for SDKs that predate the filesystem API.
            filesystem = getattr(sandbox, "filesystem", None)
            if filesystem is not None:
                await call_maybe_async(filesystem.write_bytes, data, path)
            else:
                handle_file = await call_maybe_async(sandbox.open, path, "wb")
                await call_maybe_async(handle_file.write, data)
                await call_maybe_async(handle_file.close)
            return SandboxFileResult(ok=True, evidence=SandboxEvidence.from_bytes("file.write", data, metadata={"path": path}))
        except Exception as error:
            return SandboxFileResult(ok=False, error=ContractError(code="sandbox_file_write_failed", message=safe_exception_message(error)))

    async def read_file(self, handle: SandboxHandle, path: str) -> SandboxFileResult:
        try:
            sandbox = await self._resolve(handle)
            filesystem = getattr(sandbox, "filesystem", None)
            if filesystem is not None:
                data = to_bytes(await call_maybe_async(filesystem.read_bytes, path))
            else:
                handle_file = await call_maybe_async(sandbox.open, path, "rb")
                data = to_bytes(await call_maybe_async(handle_file.read))
                await call_maybe_async(handle_file.close)
            return SandboxFileResult(ok=True, data=data, evidence=SandboxEvidence.from_bytes("file.read", data, metadata={"path": path}))
        except Exception as error:
            return SandboxFileResult(ok=False, error=ContractError(code="sandbox_file_read_failed", message=safe_exception_message(error)))

    async def cancel(self, handle: SandboxHandle) -> SandboxActionResult:
        try:
            sandbox = await self._resolve(handle)
            await call_maybe_async(sandbox.terminate)
            return SandboxActionResult(ok=True, evidence=SandboxEvidence.from_bytes("sandbox.cancel", handle.id.encode()))
        except Exception as error:
            return SandboxActionResult(ok=False, error=ContractError(code="sandbox_cancel_failed", message=safe_exception_message(error)))

    async def snapshot(self, handle: SandboxHandle, name: str) -> SandboxActionResult:
        try:
            sandbox = await self._resolve(handle)
            image = await call_maybe_async(sandbox.snapshot_filesystem)
            reference = str(getattr(image, "object_id", None) or image)
            evidence = SandboxEvidence.from_bytes("sandbox.snapshot", reference.encode(), metadata={"name": name})
            return SandboxActionResult(ok=True, reference=reference, evidence=evidence)
        except Exception as error:
            return SandboxActionResult(ok=False, error=ContractError(code="sandbox_snapshot_failed", message=safe_exception_message(error)))

    async def destroy(self, handle: SandboxHandle) -> SandboxActionResult:
        try:
            sandbox = await self._resolve(handle)
            await call_maybe_async(sandbox.terminate)
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
            provider="modal",
            isolation="container",
            networking="allowlist",
            persistent_filesystem=True,
            snapshots=True,
            command_streaming=False,
            cancellation=True,
            file_transfer=True,
            evidence=True,
            limitations=(
                "gVisor container isolation, not a dedicated VM per sandbox.",
                "Network allowlists are CIDR-only; domain allowlists fail closed.",
                "Command streaming is buffered: output arrives after the command completes.",
                "Snapshots capture the filesystem as a Modal image reference, not a running-process checkpoint.",
            ),
        )
