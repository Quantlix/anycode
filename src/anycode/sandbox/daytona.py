"""Independent Daytona sandbox provider adapter."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Callable
from typing import Any

from anycode.contracts.models import ContractError
from anycode.helpers.uuid7 import uuid7
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


class DaytonaSandboxProvider:
    """Daytona SDK adapter; credentials remain in SDK configuration or secret references."""

    def __init__(self, client: Any | None = None, *, session_request_factory: Callable[..., Any] | None = None) -> None:
        self._client = client
        self._session_request_factory = session_request_factory
        self._sandboxes: dict[str, Any] = {}

    def _client_or_raise(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from daytona import AsyncDaytona  # type: ignore[import-not-found]
        except ImportError as error:
            raise ImportError("DaytonaSandboxProvider requires the optional 'daytona' package.") from error
        self._client = AsyncDaytona()
        return self._client

    def _creation_params(self, spec: SandboxSpec) -> Any | None:
        if not any((spec.image, spec.snapshot, spec.secret_references, spec.labels, spec.allowed_domains, spec.allowed_cidrs)):
            return None
        try:
            from daytona import CreateSandboxFromImageParams, CreateSandboxFromSnapshotParams  # type: ignore[import-not-found]
        except ImportError as error:
            raise ImportError("The installed Daytona SDK does not expose sandbox creation models.") from error
        common = {
            "language": spec.language,
            "labels": {
                **spec.labels,
                "anycode.run_id": spec.run_id,
                "anycode.tenant": spec.context.tenant_scope,
                "anycode.classification": spec.context.classification,
            },
            "public": False,
            "ephemeral": not spec.persistent,
            "secrets": {name: reference.removeprefix("daytona:") for name, reference in spec.secret_references.items()},
            "network_block_all": spec.network == "none",
            "network_allow_list": ",".join(spec.allowed_cidrs) or None,
            "domain_allow_list": ",".join(spec.allowed_domains) or None,
        }
        common = {key: value for key, value in common.items() if value not in (None, {}, "")}
        if spec.image:
            return CreateSandboxFromImageParams(image=spec.image, **common)
        return CreateSandboxFromSnapshotParams(snapshot=spec.snapshot, **common)

    async def _resolve(self, handle: SandboxHandle) -> Any:
        sandbox = self._sandboxes.get(handle.id)
        if sandbox is not None:
            return sandbox
        sandbox = await self._client_or_raise().get(handle.id)
        self._sandboxes[handle.id] = sandbox
        return sandbox

    async def create(self, spec: SandboxSpec) -> SandboxCreateResult:
        try:
            client = self._client_or_raise()
            params = self._creation_params(spec)
            sandbox = await client.create(params) if params is not None else await client.create()
            sandbox_id = str(sandbox.id)
            self._sandboxes[sandbox_id] = sandbox
            handle = SandboxHandle(
                id=sandbox_id,
                provider="daytona",
                run_id=spec.run_id,
                task_id=spec.task_id,
                correlation_id=spec.correlation_id,
                context=spec.context,
                capabilities=self.capabilities(),
                metadata={"snapshot": str(getattr(sandbox, "snapshot", "") or "")},
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
            response = await sandbox.process.exec(
                command.command,
                cwd=command.cwd,
                env=command.environment or None,
                timeout=command.timeout_seconds,
            )
            stdout = str(getattr(getattr(response, "artifacts", None), "stdout", None) or getattr(response, "result", "") or "")
            stderr = str(getattr(response, "stderr", "") or "")
            exit_code = getattr(response, "exit_code", 0)
            evidence = SandboxEvidence.from_bytes("command", f"{stdout}\0{stderr}".encode(), metadata={"exit_code": exit_code})
            return SandboxCommandResult(ok=exit_code == 0, exit_code=exit_code, stdout=stdout, stderr=stderr, evidence=evidence)
        except Exception as error:
            return SandboxCommandResult(
                ok=False,
                error=ContractError(code="sandbox_command_failed", message=safe_exception_message(error), retryable=True),
            )

    def stream(self, handle: SandboxHandle, command: SandboxCommand) -> AsyncIterable[SandboxOutputChunk]:
        async def generate() -> AsyncIterator[SandboxOutputChunk]:
            sandbox = await self._resolve(handle)
            session_id = f"anycode-{uuid7()}"
            sequence = 0
            queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
            try:
                if self._session_request_factory is None:
                    from daytona import SessionExecuteRequest  # type: ignore[import-not-found]

                    request_factory = SessionExecuteRequest
                else:
                    request_factory = self._session_request_factory

                await sandbox.process.create_session(session_id)
                response = await sandbox.process.execute_session_command(
                    session_id,
                    request_factory(command=command.command, run_async=True),
                    timeout=command.timeout_seconds,
                )

                async def stdout(chunk: str) -> None:
                    await queue.put(("stdout", chunk))

                async def stderr(chunk: str) -> None:
                    await queue.put(("stderr", chunk))

                async def collect() -> None:
                    try:
                        await sandbox.process.get_session_command_logs_async(session_id, response.cmd_id, stdout, stderr)
                    finally:
                        await queue.put(None)

                collector = asyncio.create_task(collect())
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    sequence += 1
                    yield SandboxOutputChunk(stream=item[0], data=item[1], sequence=sequence)  # type: ignore[arg-type]
                await collector
            finally:
                try:
                    await sandbox.process.delete_session(session_id)
                except Exception:
                    pass

        return generate()

    async def write_file(self, handle: SandboxHandle, path: str, data: bytes) -> SandboxFileResult:
        try:
            sandbox = await self._resolve(handle)
            await sandbox.fs.upload_file(data, path)
            return SandboxFileResult(ok=True, evidence=SandboxEvidence.from_bytes("file.write", data, metadata={"path": path}))
        except Exception as error:
            return SandboxFileResult(ok=False, error=ContractError(code="sandbox_file_write_failed", message=safe_exception_message(error)))

    async def read_file(self, handle: SandboxHandle, path: str) -> SandboxFileResult:
        try:
            sandbox = await self._resolve(handle)
            data = bytes(await sandbox.fs.download_file(path))
            return SandboxFileResult(ok=True, data=data, evidence=SandboxEvidence.from_bytes("file.read", data, metadata={"path": path}))
        except Exception as error:
            return SandboxFileResult(ok=False, error=ContractError(code="sandbox_file_read_failed", message=safe_exception_message(error)))

    async def cancel(self, handle: SandboxHandle) -> SandboxActionResult:
        try:
            sandbox = await self._resolve(handle)
            if hasattr(sandbox, "stop"):
                await sandbox.stop(force=True)
            else:
                await self._client_or_raise().stop(sandbox)
            return SandboxActionResult(ok=True, evidence=SandboxEvidence.from_bytes("sandbox.cancel", handle.id.encode()))
        except Exception as error:
            return SandboxActionResult(ok=False, error=ContractError(code="sandbox_cancel_failed", message=safe_exception_message(error)))

    async def snapshot(self, handle: SandboxHandle, name: str) -> SandboxActionResult:
        del handle, name
        return SandboxActionResult(
            ok=False,
            error=ContractError(
                code="sandbox_snapshot_unsupported",
                message="Daytona point-in-time sandbox snapshots are not in the stable SDK surface.",
            ),
        )

    async def destroy(self, handle: SandboxHandle) -> SandboxActionResult:
        try:
            sandbox = await self._resolve(handle)
            if hasattr(sandbox, "delete"):
                await sandbox.delete()
            else:
                await self._client_or_raise().delete(sandbox)
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
            provider="daytona",
            isolation="remote",
            networking="allowlist",
            persistent_filesystem=True,
            snapshots=False,
            command_streaming=True,
            cancellation=True,
            file_transfer=True,
            evidence=True,
            limitations=(
                "Isolation strength and placement depend on the selected Daytona runner and sandbox class.",
                "Cancellation force-stops the sandbox rather than only one command.",
                "Point-in-time sandbox snapshots are not exposed through the stable adapter.",
            ),
        )
