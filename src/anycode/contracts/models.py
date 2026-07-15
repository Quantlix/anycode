"""Versioned, language-neutral semantic contract models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

CONTRACT_SCHEMA_VERSION: Literal["1.0"] = "1.0"
EVENT_PAYLOAD_VERSION = 1

RunState = Literal["accepted", "queued", "running", "waiting", "succeeded", "failed", "canceled", "rejected"]
WaitingReason = Literal[
    "dependency",
    "schedule",
    "input_required",
    "authorization_required",
    "approval_required",
    "retry_backoff",
    "capacity",
    "external_signal",
]
CancellationStatus = Literal["none", "requested", "acknowledged", "lost_to_completion"]
ArtifactClassification = Literal["public", "internal", "confidential", "restricted"]


def utc_now() -> datetime:
    return datetime.now(UTC)


class ContractModel(BaseModel):
    """Strict immutable base shared by all wire-contract models."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ContractError(ContractModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False
    details: dict[str, JsonValue] = Field(default_factory=dict)


class Result(ContractModel):
    ok: bool
    value: JsonValue | None = None
    error: ContractError | None = None

    @model_validator(mode="after")
    def _validate_result(self) -> Result:
        if self.ok and self.error is not None:
            raise ValueError("A successful result cannot contain an error.")
        if not self.ok and self.error is None:
            raise ValueError("A failed result requires an error.")
        return self


class Cancellation(ContractModel):
    status: CancellationStatus = "none"
    requested_at: datetime | None = None
    acknowledged_at: datetime | None = None
    reason: str | None = None


class Run(ContractModel):
    schema_version: Literal["1.0"] = CONTRACT_SCHEMA_VERSION
    id: str = Field(min_length=1)
    state: RunState = "accepted"
    root_task_id: str | None = None
    correlation_id: str = Field(min_length=1)
    causation_id: str | None = None
    generation: int = Field(default=1, ge=1)
    attempt: int = Field(default=1, ge=1)
    waiting_reason: WaitingReason | None = None
    cancellation: Cancellation = Field(default_factory=Cancellation)
    error: ContractError | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    last_event_sequence: int = Field(default=0, ge=0)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_waiting_reason(self) -> Run:
        if self.state == "waiting" and self.waiting_reason is None:
            raise ValueError("A waiting run requires waiting_reason.")
        if self.state != "waiting" and self.waiting_reason is not None:
            raise ValueError("waiting_reason is only valid for a waiting run.")
        return self


class Task(ContractModel):
    schema_version: Literal["1.0"] = CONTRACT_SCHEMA_VERSION
    id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    state: RunState = "accepted"
    title: str = Field(min_length=1)
    description: str = ""
    correlation_id: str = Field(min_length=1)
    causation_id: str | None = None
    generation: int = Field(default=1, ge=1)
    attempt: int = Field(default=1, ge=1)
    dependencies: tuple[str, ...] = ()
    waiting_reason: WaitingReason | None = None
    produced_artifact_ids: tuple[str, ...] = ()
    allow_partial_dependency_artifacts: bool = False
    error: ContractError | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_waiting_reason(self) -> Task:
        if self.state == "waiting" and self.waiting_reason is None:
            raise ValueError("A waiting task requires waiting_reason.")
        if self.state != "waiting" and self.waiting_reason is not None:
            raise ValueError("waiting_reason is only valid for a waiting task.")
        if self.id in self.dependencies:
            raise ValueError("A task cannot depend on itself.")
        return self


class TextPart(ContractModel):
    type: Literal["text"] = "text"
    text: str


class DataPart(ContractModel):
    type: Literal["data"] = "data"
    data: JsonValue


class ArtifactPart(ContractModel):
    type: Literal["artifact"] = "artifact"
    artifact_id: str = Field(min_length=1)


MessagePart = Annotated[TextPart | DataPart | ArtifactPart, Field(discriminator="type")]


class Message(ContractModel):
    schema_version: Literal["1.0"] = CONTRACT_SCHEMA_VERSION
    id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str | None = None
    role: Literal["system", "user", "agent", "tool"]
    parts: tuple[MessagePart, ...]
    correlation_id: str = Field(min_length=1)
    causation_id: str | None = None
    generation: int = Field(default=1, ge=1)
    attempt: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class InlineArtifactContent(ContractModel):
    form: Literal["inline"] = "inline"
    data: str
    encoding: Literal["utf-8", "base64"] = "base64"


class ArtifactReference(ContractModel):
    form: Literal["reference"] = "reference"
    uri: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    expires_at: datetime | None = None


ArtifactContent = Annotated[InlineArtifactContent | ArtifactReference, Field(discriminator="form")]


class ArtifactProvenance(ContractModel):
    producer: str = Field(min_length=1)
    source_artifact_ids: tuple[str, ...] = ()
    operation_key: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class ArtifactRetention(ContractModel):
    retain_until: datetime | None = None
    legal_hold: bool = False
    policy: str | None = None


class Artifact(ContractModel):
    schema_version: Literal["1.0"] = CONTRACT_SCHEMA_VERSION
    id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str | None = None
    name: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    size: int = Field(ge=0)
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    content: ArtifactContent
    provenance: ArtifactProvenance
    classification: ArtifactClassification = "internal"
    retention: ArtifactRetention = Field(default_factory=ArtifactRetention)
    correlation_id: str = Field(min_length=1)
    causation_id: str | None = None
    generation: int = Field(default=1, ge=1)
    attempt: int = Field(default=1, ge=1)
    finalized: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class Event(ContractModel):
    schema_version: Literal["1.0"] = CONTRACT_SCHEMA_VERSION
    payload_version: int = Field(default=EVENT_PAYLOAD_VERSION, ge=1)
    id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str | None = None
    sequence: int = Field(ge=1)
    type: str = Field(min_length=1)
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    correlation_id: str = Field(min_length=1)
    causation_id: str | None = None
    generation: int = Field(default=1, ge=1)
    attempt: int = Field(default=1, ge=1)
    emitted_at: datetime = Field(default_factory=utc_now)


class Checkpoint(ContractModel):
    schema_version: Literal["1.0"] = CONTRACT_SCHEMA_VERSION
    id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    event_cursor: int = Field(ge=0)
    generation: int = Field(ge=1)
    attempt: int = Field(ge=1)
    correlation_id: str = Field(min_length=1)
    causation_id: str | None = None
    run: Run
    tasks: tuple[Task, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_identity(self) -> Checkpoint:
        if self.run_id != self.run.id:
            raise ValueError("Checkpoint run_id must match the embedded run id.")
        if self.generation != self.run.generation:
            raise ValueError("Checkpoint generation must match the embedded run generation.")
        if self.attempt != self.run.attempt:
            raise ValueError("Checkpoint attempt must match the embedded run attempt.")
        if self.correlation_id != self.run.correlation_id:
            raise ValueError("Checkpoint correlation_id must match the embedded run correlation_id.")
        if self.event_cursor != self.run.last_event_sequence:
            raise ValueError("Checkpoint cursor must match the embedded run event sequence.")
        if any(task.run_id != self.run_id for task in self.tasks):
            raise ValueError("Every checkpoint task must belong to the checkpoint run.")
        return self


class PolicyObligation(ContractModel):
    type: str = Field(min_length=1)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class PolicyDecision(ContractModel):
    schema_version: Literal["1.0"] = CONTRACT_SCHEMA_VERSION
    id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str | None = None
    outcome: Literal["allow", "deny"]
    policy_version: str = Field(min_length=1)
    reason_codes: tuple[str, ...] = ()
    obligations: tuple[PolicyObligation, ...] = ()
    correlation_id: str = Field(min_length=1)
    causation_id: str | None = None
    generation: int = Field(default=1, ge=1)
    attempt: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)


class VerificationResult(ContractModel):
    schema_version: Literal["1.0"] = CONTRACT_SCHEMA_VERSION
    id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str | None = None
    verifier: str = Field(min_length=1)
    passed: bool
    severity: Literal["info", "warning", "error", "critical"]
    message: str
    evidence: dict[str, JsonValue] = Field(default_factory=dict)
    correlation_id: str = Field(min_length=1)
    causation_id: str | None = None
    generation: int = Field(default=1, ge=1)
    attempt: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)


class CapabilityDescriptor(ContractModel):
    schema_version: Literal["1.0"] = CONTRACT_SCHEMA_VERSION
    name: str = Field(min_length=1)
    implementation_version: str = Field(min_length=1)
    contract_versions: tuple[str, ...] = (CONTRACT_SCHEMA_VERSION,)
    operations: tuple[str, ...] = ()
    artifact_forms: tuple[Literal["inline", "reference"], ...] = ("inline", "reference")
    max_inline_artifact_bytes: int = Field(default=65_536, ge=0)
    supports_cancellation: bool = True
    supports_resume: bool = True
    supports_event_stream: bool = True
    extensions: dict[str, JsonValue] = Field(default_factory=dict)


CONTRACT_MODELS: tuple[type[ContractModel], ...] = (
    Run,
    Task,
    Message,
    Artifact,
    Event,
    Checkpoint,
    PolicyDecision,
    VerificationResult,
    CapabilityDescriptor,
)
