"""Immutable models shared by durability backend implementations."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from anycode.contracts.models import ArtifactReference, Checkpoint, ContractError, ContractModel, Event, Run, Task, utc_now
from anycode.identity.context import ExecutionContext

BACKEND_CONTRACT_VERSION = "1.0"


class BackendCapabilities(ContractModel):
    """Guarantees exposed by a durability backend and its configured store."""

    backend: str = Field(min_length=1)
    contract_version: Literal["1.0"] = BACKEND_CONTRACT_VERSION
    persistent: bool
    external: bool
    atomic_admission: bool = True
    optimistic_concurrency: bool = True
    leases: bool = True
    fencing: bool = True
    checkpoints: bool = True
    durable_wakes: bool = True
    external_signals: bool = True
    artifact_references: bool = True
    limitations: tuple[str, ...] = ()


class BackendVersion(ContractModel):
    backend: str = Field(min_length=1)
    implementation_version: str = Field(min_length=1)
    contract_version: Literal["1.0"] = BACKEND_CONTRACT_VERSION
    store_name: str | None = None
    store_version: str | None = None


class BackendHealth(ContractModel):
    status: Literal["healthy", "degraded", "unavailable"]
    checked_at: datetime = Field(default_factory=utc_now)
    message: str = ""
    details: dict[str, JsonValue] = Field(default_factory=dict)


class Admission(ContractModel):
    admission_key: str = Field(min_length=1)
    run: Run
    initial_event: Event
    tasks: tuple[Task, ...] = ()

    @model_validator(mode="after")
    def _validate_admission(self) -> Admission:
        if self.initial_event.run_id != self.run.id or self.initial_event.sequence != 1:
            raise ValueError("The initial event must be sequence 1 and belong to the admitted run.")
        if self.run.last_event_sequence not in (0, 1):
            raise ValueError("An admitted run must have event cursor 0 or 1.")
        if any(task.run_id != self.run.id for task in self.tasks):
            raise ValueError("Every admitted task must belong to the admitted run.")
        return self


class AdmissionResult(ContractModel):
    admitted: bool
    duplicate: bool = False
    run: Run | None = None
    error: ContractError | None = None


class WorkItem(ContractModel):
    id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    generation: int = Field(default=1, ge=1)
    priority: int = 0
    available_at: datetime = Field(default_factory=utc_now)
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    execution_context: ExecutionContext | None = None


class WorkClaim(ContractModel):
    work: WorkItem
    owner_id: str = Field(min_length=1)
    generation: int = Field(ge=1)
    fencing_token: int = Field(ge=1)
    claimed_at: datetime
    lease_expires_at: datetime

    @model_validator(mode="after")
    def _validate_lease(self) -> WorkClaim:
        if self.lease_expires_at <= self.claimed_at:
            raise ValueError("A work claim lease must expire after it is acquired.")
        return self


class ClaimResult(ContractModel):
    claimed: bool
    claim: WorkClaim | None = None
    error: ContractError | None = None


class AppendResult(ContractModel):
    accepted: bool
    current_sequence: int = Field(ge=0)
    event: Event | None = None
    error: ContractError | None = None


class CommitResult(AppendResult):
    stale_owner: bool = False


class WakeRegistration(ContractModel):
    id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    wake_at: datetime
    reason: str = Field(min_length=1)
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class ExternalSignal(ContractModel):
    id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    payload: JsonValue = None
    delivered_at: datetime = Field(default_factory=utc_now)
    execution_context: ExecutionContext | None = None


class ArtifactReferenceRecord(ContractModel):
    artifact_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    reference: ArtifactReference
    recorded_at: datetime = Field(default_factory=utc_now)


class BackendSnapshot(ContractModel):
    """Portable export used by local-to-external migration tooling."""

    run: Run
    tasks: tuple[Task, ...] = ()
    events: tuple[Event, ...] = ()
    checkpoint: Checkpoint | None = None
    wakes: tuple[WakeRegistration, ...] = ()
    signals: tuple[ExternalSignal, ...] = ()
    artifact_references: tuple[ArtifactReferenceRecord, ...] = ()
