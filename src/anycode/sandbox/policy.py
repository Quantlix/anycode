"""Policy-enforced wrapper for any sandbox provider."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator

from anycode.contracts.models import ContractError
from anycode.identity.policy import PolicyEnforcer, PolicyRequest
from anycode.sandbox.models import (
    SandboxActionResult,
    SandboxCapabilities,
    SandboxCommand,
    SandboxCommandResult,
    SandboxCreateResult,
    SandboxFileResult,
    SandboxHandle,
    SandboxHealth,
    SandboxOutputChunk,
    SandboxSpec,
)
from anycode.sandbox.protocol import SandboxProvider


class PolicySandboxProvider:
    """Requires an allow decision and fulfilled obligations before each operation."""

    def __init__(self, provider: SandboxProvider, enforcer: PolicyEnforcer) -> None:
        self._provider = provider
        self._enforcer = enforcer

    async def _allowed(self, handle: SandboxHandle, action: str, input: dict[str, object] | None = None) -> ContractError | None:
        result = await self._enforcer.enforce(
            PolicyRequest(
                run_id=handle.run_id,
                task_id=handle.task_id,
                action=action,
                resource=f"sandbox:{handle.provider}:{handle.id}",
                boundary="sandbox",
                context=handle.context,
                correlation_id=handle.correlation_id,
                input=input or {},  # type: ignore[arg-type]
            )
        )
        return result.error if not result.allowed else None

    async def create(self, spec: SandboxSpec) -> SandboxCreateResult:
        temporary = SandboxHandle(
            id="pending",
            provider=self._provider.capabilities().provider,
            run_id=spec.run_id,
            task_id=spec.task_id,
            correlation_id=spec.correlation_id,
            context=spec.context,
            capabilities=self._provider.capabilities(),
        )
        denied = await self._allowed(temporary, "create", {"image": spec.image, "snapshot": spec.snapshot, "network": spec.network})
        if denied:
            return SandboxCreateResult(ok=False, error=denied)
        return await self._provider.create(spec)

    async def execute(self, handle: SandboxHandle, command: SandboxCommand) -> SandboxCommandResult:
        denied = await self._allowed(handle, "execute", {"command": command.command, "cwd": command.cwd})
        if denied:
            return SandboxCommandResult(ok=False, error=denied)
        return await self._provider.execute(handle, command)

    def stream(self, handle: SandboxHandle, command: SandboxCommand) -> AsyncIterable[SandboxOutputChunk]:
        async def generate() -> AsyncIterator[SandboxOutputChunk]:
            denied = await self._allowed(handle, "stream", {"command": command.command, "cwd": command.cwd})
            if denied:
                yield SandboxOutputChunk(stream="error", data=denied.message, sequence=1)
                return
            async for chunk in self._provider.stream(handle, command):
                yield chunk

        return generate()

    async def write_file(self, handle: SandboxHandle, path: str, data: bytes) -> SandboxFileResult:
        denied = await self._allowed(handle, "file.write", {"path": path, "size": len(data)})
        if denied:
            return SandboxFileResult(ok=False, error=denied)
        return await self._provider.write_file(handle, path, data)

    async def read_file(self, handle: SandboxHandle, path: str) -> SandboxFileResult:
        denied = await self._allowed(handle, "file.read", {"path": path})
        if denied:
            return SandboxFileResult(ok=False, error=denied)
        return await self._provider.read_file(handle, path)

    async def cancel(self, handle: SandboxHandle) -> SandboxActionResult:
        denied = await self._allowed(handle, "cancel")
        if denied:
            return SandboxActionResult(ok=False, error=denied)
        return await self._provider.cancel(handle)

    async def snapshot(self, handle: SandboxHandle, name: str) -> SandboxActionResult:
        denied = await self._allowed(handle, "snapshot", {"name": name})
        if denied:
            return SandboxActionResult(ok=False, error=denied)
        return await self._provider.snapshot(handle, name)

    async def destroy(self, handle: SandboxHandle) -> SandboxActionResult:
        denied = await self._allowed(handle, "destroy")
        if denied:
            return SandboxActionResult(ok=False, error=denied)
        return await self._provider.destroy(handle)

    async def health(self) -> SandboxHealth:
        return await self._provider.health()

    def capabilities(self) -> SandboxCapabilities:
        return self._provider.capabilities()
