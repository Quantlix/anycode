"""Dapr state-store durability adapter with ETag compare-and-set writes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, TypeVar, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from pydantic import JsonValue

from anycode.backends.memory import AmbiguousBackendResultError, BackendUnavailableError, InMemoryDurabilityBackend
from anycode.backends.models import (
    Admission,
    AdmissionResult,
    AppendResult,
    ArtifactReferenceRecord,
    BackendCapabilities,
    BackendHealth,
    BackendSnapshot,
    BackendVersion,
    ClaimResult,
    CommitResult,
    ExternalSignal,
    WakeRegistration,
    WorkClaim,
    WorkItem,
)
from anycode.contracts.models import Checkpoint, Event, Run, Task
from anycode.security.redaction import safe_exception_message

_IMPLEMENTATION_VERSION = "1.0"
_DEFAULT_STATE_KEY = "anycode-runtime-state-v1"
_DEFAULT_TIMEOUT_SECONDS = 10.0
_DEFAULT_CONFLICT_RETRIES = 8
T = TypeVar("T")


@dataclass(frozen=True)
class DaprStateRecord:
    value: dict[str, JsonValue] | None
    etag: str | None


@runtime_checkable
class DaprStateTransport(Protocol):
    async def get(self, key: str) -> DaprStateRecord: ...

    async def compare_and_set(self, key: str, value: dict[str, JsonValue], etag: str | None) -> bool: ...

    async def health(self) -> bool: ...


class DaprHTTPTransport:
    """Minimal stdlib transport for the Dapr v1.0 state and health APIs."""

    def __init__(
        self,
        store_name: str,
        *,
        base_url: str = "http://127.0.0.1:3500",
        api_token: str | None = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not store_name:
            raise ValueError("store_name must not be empty")
        self.store_name = store_name
        self._base_url = base_url.rstrip("/")
        self._api_token = api_token
        self._timeout_seconds = timeout_seconds

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_token:
            headers["dapr-api-token"] = self._api_token
        return headers

    def _request(self, path: str, *, method: str = "GET", payload: object | None = None) -> tuple[int, dict[str, str], bytes]:
        data = json.dumps(payload, separators=(",", ":")).encode() if payload is not None else None
        request = Request(f"{self._base_url}{path}", data=data, headers=self._headers(), method=method)
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310 - configured Dapr sidecar endpoint
                return response.status, dict(response.headers.items()), response.read()
        except HTTPError as error:
            if error.code in (204, 404, 409):
                return error.code, dict(error.headers.items()), error.read()
            raise BackendUnavailableError(f"Dapr request failed with HTTP {error.code}") from error
        except (TimeoutError, URLError, OSError) as error:
            raise BackendUnavailableError(f"Dapr sidecar is unavailable: {safe_exception_message(error)}") from error

    async def get(self, key: str) -> DaprStateRecord:
        path = f"/v1.0/state/{quote(self.store_name, safe='')}/{quote(key, safe='')}?consistency=strong"
        status, headers, body = await asyncio.to_thread(self._request, path)
        if status in (204, 404) or not body:
            return DaprStateRecord(value=None, etag=None)
        value = json.loads(body)
        if not isinstance(value, dict):
            raise BackendUnavailableError("Dapr state payload is not a JSON object")
        etag = next((value for name, value in headers.items() if name.casefold() == "etag"), None)
        return DaprStateRecord(value=value, etag=etag)

    async def compare_and_set(self, key: str, value: dict[str, JsonValue], etag: str | None) -> bool:
        request_payload: dict[str, object] = {
            "key": key,
            "value": value,
            "options": {"concurrency": "first-write", "consistency": "strong"},
        }
        if etag is not None:
            request_payload["etag"] = etag
        transaction = {"operations": [{"operation": "upsert", "request": request_payload}]}
        path = f"/v1.0/state/{quote(self.store_name, safe='')}/transaction"
        status, _, _ = await asyncio.to_thread(self._request, path, method="POST", payload=transaction)
        return status != 409

    async def health(self) -> bool:
        try:
            status, _, _ = await asyncio.to_thread(self._request, "/v1.0/healthz/outbound")
        except BackendUnavailableError:
            return False
        return 200 <= status < 300


class DaprDurabilityBackend(InMemoryDurabilityBackend):
    """External backend backed by a transactional, ETag-capable Dapr state store."""

    def __init__(
        self,
        transport: DaprStateTransport,
        *,
        state_key: str = _DEFAULT_STATE_KEY,
        max_conflict_retries: int = _DEFAULT_CONFLICT_RETRIES,
    ) -> None:
        super().__init__()
        if max_conflict_retries < 1:
            raise ValueError("max_conflict_retries must be at least 1")
        self._transport = transport
        self._state_key = state_key
        self._max_conflict_retries = max_conflict_retries
        self._remote_lock = asyncio.Lock()

    async def _with_state(self, operation: Callable[[], Awaitable[T]], *, write: bool) -> T:
        async with self._remote_lock:
            for attempt in range(self._max_conflict_retries):
                record = await self._transport.get(self._state_key)
                self._restore_state(record.value)
                try:
                    result = await operation()
                except AmbiguousBackendResultError:
                    if write and await self._transport.compare_and_set(self._state_key, self._dump_state(), record.etag):
                        raise
                    continue
                if not write:
                    return result
                if await self._transport.compare_and_set(self._state_key, self._dump_state(), record.etag):
                    return result
                if attempt + 1 == self._max_conflict_retries:
                    break
            raise BackendUnavailableError("Dapr state remained contended after the configured compare-and-set retries")

    async def admit(self, admission: Admission) -> AdmissionResult:
        operation = super().admit
        return await self._with_state(lambda: operation(admission), write=True)

    async def enqueue(self, work: WorkItem) -> None:
        operation = super().enqueue
        await self._with_state(lambda: operation(work), write=True)

    async def claim(self, owner_id: str, *, lease_seconds: float = 30.0) -> ClaimResult:
        operation = super().claim
        return await self._with_state(lambda: operation(owner_id, lease_seconds=lease_seconds), write=True)

    async def heartbeat(self, claim: WorkClaim, *, lease_seconds: float = 30.0) -> ClaimResult:
        operation = super().heartbeat
        return await self._with_state(lambda: operation(claim, lease_seconds=lease_seconds), write=True)

    async def append_event(
        self,
        event: Event,
        *,
        expected_sequence: int,
        run: Run | None = None,
        tasks: tuple[Task, ...] = (),
    ) -> AppendResult:
        operation = super().append_event
        return await self._with_state(lambda: operation(event, expected_sequence=expected_sequence, run=run, tasks=tasks), write=True)

    async def commit(
        self,
        claim: WorkClaim,
        event: Event,
        *,
        expected_sequence: int,
        run: Run | None = None,
        task: Task | None = None,
    ) -> CommitResult:
        operation = super().commit
        return await self._with_state(lambda: operation(claim, event, expected_sequence=expected_sequence, run=run, task=task), write=True)

    async def request_cancellation(self, run: Run, event: Event, *, expected_sequence: int) -> AppendResult:
        operation = super().request_cancellation
        return await self._with_state(lambda: operation(run, event, expected_sequence=expected_sequence), write=True)

    async def save_checkpoint(self, checkpoint: Checkpoint) -> AppendResult:
        operation = super().save_checkpoint
        return await self._with_state(lambda: operation(checkpoint), write=True)

    async def load_checkpoint(self, run_id: str) -> Checkpoint | None:
        operation = super().load_checkpoint
        return await self._with_state(lambda: operation(run_id), write=False)

    async def register_wake(self, wake: WakeRegistration) -> None:
        operation = super().register_wake
        await self._with_state(lambda: operation(wake), write=True)

    async def due_wakes(self, *, before: datetime | None = None) -> tuple[WakeRegistration, ...]:
        operation = super().due_wakes
        return await self._with_state(lambda: operation(before=before), write=False)

    async def deliver_signal(self, signal: ExternalSignal) -> bool:
        operation = super().deliver_signal
        return await self._with_state(lambda: operation(signal), write=True)

    async def read_signals(self, run_id: str) -> tuple[ExternalSignal, ...]:
        operation = super().read_signals
        return await self._with_state(lambda: operation(run_id), write=False)

    async def read_events(self, run_id: str, *, after_sequence: int = 0) -> tuple[Event, ...]:
        operation = super().read_events
        return await self._with_state(lambda: operation(run_id, after_sequence=after_sequence), write=False)

    async def record_artifact_reference(self, record: ArtifactReferenceRecord) -> None:
        operation = super().record_artifact_reference
        await self._with_state(lambda: operation(record), write=True)

    async def read_artifact_references(self, run_id: str) -> tuple[ArtifactReferenceRecord, ...]:
        operation = super().read_artifact_references
        return await self._with_state(lambda: operation(run_id), write=False)

    async def export_run(self, run_id: str) -> BackendSnapshot | None:
        operation = super().export_run
        return await self._with_state(lambda: operation(run_id), write=False)

    async def health(self) -> BackendHealth:
        healthy = await self._transport.health()
        return BackendHealth(
            status="healthy" if healthy else "unavailable",
            message="" if healthy else "Dapr sidecar or configured state store is unavailable.",
        )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend="dapr",
            persistent=True,
            external=True,
            limitations=(
                "The configured Dapr state store must support transactions, strong reads, and ETags.",
                "This preview stores one coarse-grained runtime aggregate; store item-size and contention limits apply.",
            ),
        )

    def version(self) -> BackendVersion:
        store_name = getattr(self._transport, "store_name", None)
        return BackendVersion(backend="dapr", implementation_version=_IMPLEMENTATION_VERSION, store_name=store_name)
