"""Adapter for the separately deployed AnyCode sandbox companion."""

from __future__ import annotations

from collections.abc import AsyncIterable
from typing import Protocol, runtime_checkable

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


@runtime_checkable
class CompanionSandboxClient(Protocol):
    async def create(self, spec: SandboxSpec) -> SandboxCreateResult: ...

    async def execute(self, handle: SandboxHandle, command: SandboxCommand) -> SandboxCommandResult: ...

    def stream(self, handle: SandboxHandle, command: SandboxCommand) -> AsyncIterable[SandboxOutputChunk]: ...

    async def write_file(self, handle: SandboxHandle, path: str, data: bytes) -> SandboxFileResult: ...

    async def read_file(self, handle: SandboxHandle, path: str) -> SandboxFileResult: ...

    async def cancel(self, handle: SandboxHandle) -> SandboxActionResult: ...

    async def snapshot(self, handle: SandboxHandle, name: str) -> SandboxActionResult: ...

    async def destroy(self, handle: SandboxHandle) -> SandboxActionResult: ...

    async def health(self) -> SandboxHealth: ...


class CompanionSandboxAdapter:
    """Thin dependency-direction adapter; core never imports the companion SDK."""

    def __init__(self, client: CompanionSandboxClient, capabilities: SandboxCapabilities) -> None:
        self._client = client
        self._capabilities = capabilities

    async def create(self, spec: SandboxSpec) -> SandboxCreateResult:
        return await self._client.create(spec)

    async def execute(self, handle: SandboxHandle, command: SandboxCommand) -> SandboxCommandResult:
        return await self._client.execute(handle, command)

    def stream(self, handle: SandboxHandle, command: SandboxCommand) -> AsyncIterable[SandboxOutputChunk]:
        return self._client.stream(handle, command)

    async def write_file(self, handle: SandboxHandle, path: str, data: bytes) -> SandboxFileResult:
        return await self._client.write_file(handle, path, data)

    async def read_file(self, handle: SandboxHandle, path: str) -> SandboxFileResult:
        return await self._client.read_file(handle, path)

    async def cancel(self, handle: SandboxHandle) -> SandboxActionResult:
        return await self._client.cancel(handle)

    async def snapshot(self, handle: SandboxHandle, name: str) -> SandboxActionResult:
        return await self._client.snapshot(handle, name)

    async def destroy(self, handle: SandboxHandle) -> SandboxActionResult:
        return await self._client.destroy(handle)

    async def health(self) -> SandboxHealth:
        return await self._client.health()

    def capabilities(self) -> SandboxCapabilities:
        return self._capabilities
