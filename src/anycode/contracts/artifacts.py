"""Provider-neutral artifact contract and local reference store."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import Field, JsonValue

from anycode.contracts.models import (
    Artifact,
    ArtifactClassification,
    ArtifactProvenance,
    ArtifactReference,
    ArtifactRetention,
    ContractError,
    ContractModel,
    InlineArtifactContent,
)

DEFAULT_MAX_INLINE_ARTIFACT_BYTES = 65_536
ARTIFACT_PROVIDER_NAME = "anycode-local"


class ArtifactAccessContext(ContractModel):
    principal: str = Field(min_length=1)
    scopes: tuple[str, ...] = ()
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class ArtifactWriteRequest(ContractModel):
    artifact_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str | None = None
    name: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    data: str
    encoding: Literal["utf-8", "base64"] = "base64"
    producer: str = Field(min_length=1)
    source_artifact_ids: tuple[str, ...] = ()
    operation_key: str | None = None
    classification: ArtifactClassification = "internal"
    retention: ArtifactRetention = Field(default_factory=ArtifactRetention)
    correlation_id: str = Field(min_length=1)
    causation_id: str | None = None
    generation: int = Field(default=1, ge=1)
    attempt: int = Field(default=1, ge=1)
    inline: bool | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    def decode(self) -> bytes:
        if self.encoding == "utf-8":
            return self.data.encode("utf-8")
        return base64.b64decode(self.data, validate=True)


class ArtifactWriteResult(ContractModel):
    ok: bool
    artifact: Artifact | None = None
    error: ContractError | None = None


class ArtifactReadResult(ContractModel):
    ok: bool
    artifact: Artifact | None = None
    data_base64: str | None = None
    error: ContractError | None = None

    def read_bytes(self) -> bytes | None:
        return base64.b64decode(self.data_base64) if self.data_base64 is not None else None


ArtifactAccessHook = Callable[[Literal["read", "write"], Artifact, ArtifactAccessContext], bool | Awaitable[bool]]


@runtime_checkable
class ArtifactStore(Protocol):
    async def put(self, request: ArtifactWriteRequest, context: ArtifactAccessContext) -> ArtifactWriteResult: ...

    async def get(self, artifact_id: str, context: ArtifactAccessContext) -> ArtifactReadResult: ...


def _digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


class LocalArtifactStore:
    """Atomic content-addressed store supporting inline and referenced artifacts."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_inline_bytes: int = DEFAULT_MAX_INLINE_ARTIFACT_BYTES,
        access_hook: ArtifactAccessHook | None = None,
    ) -> None:
        if max_inline_bytes < 0:
            raise ValueError("max_inline_bytes must be non-negative.")
        self._root = Path(root)
        self._max_inline_bytes = max_inline_bytes
        self._access_hook = access_hook
        self._lock = asyncio.Lock()

    @property
    def root(self) -> Path:
        return self._root

    async def put(self, request: ArtifactWriteRequest, context: ArtifactAccessContext) -> ArtifactWriteResult:
        try:
            data = request.decode()
        except (ValueError, UnicodeError) as error:
            return ArtifactWriteResult(
                ok=False,
                error=ContractError(code="invalid_artifact_encoding", message=f"Artifact data could not be decoded: {error}"),
            )

        digest = _digest(data)
        use_inline = request.inline if request.inline is not None else len(data) <= self._max_inline_bytes
        if use_inline:
            content = InlineArtifactContent(data=base64.b64encode(data).decode("ascii"), encoding="base64")
        else:
            content = ArtifactReference(uri=f"artifact://sha256/{digest.removeprefix('sha256:')}", provider=ARTIFACT_PROVIDER_NAME)
        artifact = Artifact(
            id=request.artifact_id,
            run_id=request.run_id,
            task_id=request.task_id,
            name=request.name,
            media_type=request.media_type,
            size=len(data),
            digest=digest,
            content=content,
            provenance=ArtifactProvenance(
                producer=request.producer,
                source_artifact_ids=request.source_artifact_ids,
                operation_key=request.operation_key,
            ),
            classification=request.classification,
            retention=request.retention,
            correlation_id=request.correlation_id,
            causation_id=request.causation_id,
            generation=request.generation,
            attempt=request.attempt,
            metadata=request.metadata,
        )
        if not await self._authorized("write", artifact, context):
            return ArtifactWriteResult(
                ok=False,
                error=ContractError(code="artifact_access_denied", message="Artifact write was denied by the application access hook."),
            )

        metadata_path = self._metadata_path(artifact.id)
        async with self._lock:
            if await asyncio.to_thread(metadata_path.exists):
                existing = await self._load(artifact.id)
                if existing is None or existing.digest != artifact.digest:
                    return ArtifactWriteResult(
                        ok=False,
                        error=ContractError(code="artifact_conflict", message="Artifact id already exists with different content."),
                    )
                return ArtifactWriteResult(ok=True, artifact=existing)
            if isinstance(content, ArtifactReference):
                await asyncio.to_thread(_atomic_write, self._blob_path(digest), data)
            serialized = artifact.model_dump_json(indent=2).encode("utf-8")
            await asyncio.to_thread(_atomic_write, metadata_path, serialized)
        return ArtifactWriteResult(ok=True, artifact=artifact)

    async def get(self, artifact_id: str, context: ArtifactAccessContext) -> ArtifactReadResult:
        artifact = await self._load(artifact_id)
        if artifact is None:
            return ArtifactReadResult(
                ok=False,
                error=ContractError(code="artifact_not_found", message="Artifact metadata was not found."),
            )
        if not await self._authorized("read", artifact, context):
            return ArtifactReadResult(
                ok=False,
                error=ContractError(code="artifact_access_denied", message="Artifact read was denied by the application access hook."),
            )

        try:
            if isinstance(artifact.content, InlineArtifactContent):
                if artifact.content.encoding == "utf-8":
                    data = artifact.content.data.encode("utf-8")
                else:
                    data = base64.b64decode(artifact.content.data, validate=True)
            else:
                data = await asyncio.to_thread(self._blob_path(artifact.digest).read_bytes)
        except (OSError, ValueError) as error:
            return ArtifactReadResult(
                ok=False,
                artifact=artifact,
                error=ContractError(code="artifact_unavailable", message=f"Artifact content could not be read: {error}"),
            )
        if len(data) != artifact.size or _digest(data) != artifact.digest:
            return ArtifactReadResult(
                ok=False,
                artifact=artifact,
                error=ContractError(code="artifact_integrity_failed", message="Artifact size or digest does not match its metadata."),
            )
        return ArtifactReadResult(ok=True, artifact=artifact, data_base64=base64.b64encode(data).decode("ascii"))

    async def _authorized(self, action: Literal["read", "write"], artifact: Artifact, context: ArtifactAccessContext) -> bool:
        if self._access_hook is None:
            return True
        outcome = self._access_hook(action, artifact, context)
        return await outcome if isinstance(outcome, Awaitable) else outcome

    async def _load(self, artifact_id: str) -> Artifact | None:
        path = self._metadata_path(artifact_id)
        try:
            raw = await asyncio.to_thread(path.read_text, encoding="utf-8")
            return Artifact.model_validate_json(raw)
        except (OSError, ValueError):
            return None

    def _metadata_path(self, artifact_id: str) -> Path:
        safe_name = hashlib.sha256(artifact_id.encode("utf-8")).hexdigest()
        return self._root / "metadata" / f"{safe_name}.json"

    def _blob_path(self, digest: str) -> Path:
        return self._root / "blobs" / digest.removeprefix("sha256:")
