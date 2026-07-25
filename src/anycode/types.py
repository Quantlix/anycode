"""AnyCode — Pydantic models for the entire type surface."""

from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from anycode.constants import CHECKPOINT_FORMAT_VERSION, DEFAULT_PROVIDER_CAPACITY_WAIT_SECONDS, DEFAULT_PROVIDER_CONCURRENCY
from anycode.identity.context import ExecutionContext

# -- Content blocks --


class TextBlock(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool | None = None


class ImageSource(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["base64"] = "base64"
    media_type: str
    data: str


class ImageBlock(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["image"] = "image"
    source: ImageSource


class ThinkingBlock(BaseModel):
    """Extended-thinking reasoning block.

    The ``signature`` must be preserved and returned verbatim on the next
    request or the provider rejects a thinking-enabled tool-use turn.
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str = ""


class RedactedThinkingBlock(BaseModel):
    """Encrypted reasoning the provider chose to redact; opaque, pass back as-is."""

    model_config = ConfigDict(frozen=True)
    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock | ImageBlock | ThinkingBlock | RedactedThinkingBlock


# -- Conversation messages --


class LLMMessage(BaseModel):
    model_config = ConfigDict(frozen=True)
    role: Literal["user", "assistant"]
    content: list[ContentBlock]


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


# -- Execution lifecycle --

ExecutionPhase = Literal[
    "initialized",
    "planning",
    "executing",
    "observing",
    "verifying",
    "recovering",
    "completed",
    "failed",
    "cancelled",
]

StopReasonCode = Literal[
    "success",
    "max_turns",
    "max_steps",
    "budget_exceeded",
    "context_pressure",
    "tool_error",
    "verification_failed",
    "blocked_dependency",
    "user_cancelled",
    "doom_loop",
    "provider_unavailable",
    "side_effect_unknown",
    "unknown",
]


class StopReason(BaseModel):
    model_config = ConfigDict(frozen=True)
    code: StopReasonCode
    message: str
    recoverable: bool = False


class LifecycleEvent(BaseModel):
    model_config = ConfigDict(frozen=True)
    run_id: str
    agent_name: str
    task_id: str | None = None
    phase: ExecutionPhase
    stop_reason: StopReason | None = None
    metadata: dict[str, str | int | float | bool] = {}


# -- Adaptive context lifecycle --

ContextPressure = Literal["normal", "trim", "mask", "offload", "compact", "handoff"]
ContextSourceKind = Literal["instructions", "working_memory", "task_state", "external_memory", "files", "verification", "offloaded_artifact"]


class ContextArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)
    artifact_id: str
    path: str
    bytes: int
    digest: str
    head_excerpt: str
    tail_excerpt: str
    recovery_hint: str
    source_event_id: str | None = None


class ContextSource(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: ContextSourceKind
    label: str
    estimated_tokens: int
    preserved: bool = True


# -- Context engineering --

ContextSectionKind = Literal[
    "reserved_response",
    "system_instructions",
    "tool_definitions",
    "user_messages",
    "files",
    "tool_results",
    "memory_rag",
    "task_state",
    "verification",
    "offloaded_artifacts",
]

CountingConfidence = Literal["provider", "tokenizer", "heuristic"]
TokenizerStrategy = Literal["provider", "tiktoken", "heuristic"]
ContextMode = Literal["disabled", "manual", "auto"]
SectionPriority = Literal["required", "high", "medium", "low"]
SectionOverflow = Literal["trim", "summarize", "offload", "drop", "error"]


class ModelContextProfile(BaseModel):
    """Describes the effective context window and accounting hints for a model."""

    model_config = ConfigDict(frozen=True)
    provider: str
    model: str
    max_context_tokens: int | None = None
    max_output_tokens: int | None = None
    supports_prompt_cache: bool = False
    tokenizer_strategy: TokenizerStrategy = "heuristic"


class ContextSectionBudget(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: ContextSectionKind
    max_tokens: int | None = None
    priority: SectionPriority = "medium"
    overflow: SectionOverflow = "trim"


class ContextSectionUsage(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: ContextSectionKind
    estimated_tokens: int = 0
    actual_tokens: int | None = None
    included_tokens: int = 0
    percentage_of_window: float = 0.0
    strategy_applied: str | None = None


class ContextSectionInput(BaseModel):
    """First-class context payload supplied to the section-aware assembler."""

    model_config = ConfigDict(frozen=True)
    kind: ContextSectionKind
    label: str
    content: str
    preserved: bool = True


class ContextUsageReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_context_tokens: int | None = None
    reserved_response_tokens: int = 0
    used_tokens: int = 0
    available_tokens: int | None = None
    sections: tuple[ContextSectionUsage, ...] = ()
    counting_confidence: CountingConfidence = "heuristic"
    profile: ModelContextProfile | None = None


class ContextPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool = False
    max_context_tokens: int = 100_000
    trim_ratio: float = 0.65
    mask_ratio: float = 0.70
    offload_ratio: float = 0.75
    compact_ratio: float = 0.85
    handoff_ratio: float = 0.95
    auto_reset_on_handoff: bool = False
    keep_recent_messages: int = 6
    max_tool_output_tokens: int = 4000
    summary_target_tokens: int = 800
    artifact_dir: str = ".anycode/artifacts"
    redact_sensitive_data: bool = True
    preserved_task_state: dict[str, str] = {}
    preserved_verification_failures: tuple[str, ...] = ()
    provider_overrides: dict[str, ContextPolicy] = {}
    # Context engineering extensions (all additive, backward compatible).
    mode: Literal["disabled", "manual", "auto"] = "manual"
    reserved_response_tokens: int = 0
    sections: dict[ContextSectionKind, ContextSectionBudget] = {}
    model_profile: ModelContextProfile | None = None
    custom_profiles: tuple[ModelContextProfile, ...] = ()

    def for_provider(self, provider: str | None) -> ContextPolicy:
        if provider and provider in self.provider_overrides:
            return self.provider_overrides[provider]
        return self


class ContextManifest(BaseModel):
    model_config = ConfigDict(frozen=True)
    pressure: ContextPressure
    estimated_tokens: int
    max_tokens: int
    sources: list[ContextSource] = []
    offloaded: list[ContextArtifact] = []
    compaction_summary: str | None = None
    handoff_path: str | None = None
    archive_path: str | None = None
    preserved_task_state: dict[str, str] = {}
    preserved_verification_failures: tuple[str, ...] = ()
    provider: str | None = None
    # Section-aware reporting.
    usage_report: ContextUsageReport | None = None
    actual_input_tokens: int | None = None
    warnings: tuple[str, ...] = ()


# -- Verification sensors and quality gates --

VerificationKind = Literal["computational", "inferential", "hybrid"]
VerificationSeverity = Literal["info", "warning", "error", "critical"]
GateOutcome = Literal["pass", "warn", "retry", "block", "escalate"]
SensorPhase = Literal["before_tool", "after_tool", "after_task", "after_team"]


class VerificationResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    sensor_name: str
    kind: VerificationKind
    passed: bool
    severity: VerificationSeverity
    message: str
    evidence: dict[str, str | int | float | bool] = {}
    feedback_for_agent: str | None = None
    duration_ms: float = 0.0


class VerificationSensorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    kind: VerificationKind
    phases: tuple[SensorPhase, ...] = ("after_task",)
    block_on_failure: bool = False
    retry_on_failure: bool = False
    options: dict[str, str | int | float | bool] = {}


class QualityGateDecision(BaseModel):
    model_config = ConfigDict(frozen=True)
    outcome: GateOutcome
    results: tuple[VerificationResult, ...]
    message: str


# -- Harness evaluation suite --


class EvalScenario(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    description: str = ""
    prompt: str
    system_prompt: str | None = None
    provider: str | None = None
    model: str | None = None
    success_criteria: tuple[str, ...] = ()
    forbidden_substrings: tuple[str, ...] = ()
    expected_stop_reason: str | None = None
    allowed_tools: tuple[str, ...] = ()
    max_turns: int = 4
    max_tokens: int | None = None
    temperature: float | None = None
    deterministic: bool = False
    fake_responses: tuple[str, ...] = ()
    fake_tool_failures: tuple[str, ...] = ()


class EvalScenarioResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    scenario_name: str
    passed: bool
    output: str
    stop_reason_code: str | None = None
    expected_stop_reason: str | None = None
    runtime_seconds: float
    turns: int
    tool_calls: int
    token_usage: TokenUsage = TokenUsage()
    failure_reason: str | None = None
    matched_criteria: tuple[str, ...] = ()
    missing_criteria: tuple[str, ...] = ()
    cost_usd: float = 0.0
    retries: int = 0
    verification_failures: int = 0


class EvalReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    suite_name: str
    harness_variant: str
    total_scenarios: int
    passed: int
    failed: int
    total_runtime_seconds: float
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float = 0.0
    total_retries: int = 0
    total_verification_failures: int = 0
    scenario_results: tuple[EvalScenarioResult, ...]


class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    content: list[ContentBlock]
    model: str
    stop_reason: str
    usage: TokenUsage


class StreamEvent(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["text", "thinking", "tool_use", "tool_result", "done", "error", "handoff"]
    data: Any


# -- Tool definitions --


class LLMToolDef(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    data: str
    is_error: bool | None = None
    retry_safe: bool | None = None


class AgentInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    role: str
    model: str


class TeamInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    agents: list[str]


class ToolSecurityPolicy(BaseModel):
    """Least-privilege controls applied to every tool invocation.

    Empty allowlists preserve existing behavior. Set ``workspace_root`` to
    constrain filesystem tools, disable ``allow_shell`` for untrusted agents,
    and disable ``inherit_environment`` to avoid exposing process secrets.
    """

    model_config = ConfigDict(frozen=True)
    allowed_tools: tuple[str, ...] = ()
    denied_tools: tuple[str, ...] = ()
    workspace_root: str | None = None
    allowed_path_roots: tuple[str, ...] = ()
    allow_shell: bool = True
    allowed_shell_commands: tuple[str, ...] = ()
    inherit_environment: bool = True
    allowed_environment_variables: tuple[str, ...] = ()


class ToolIdempotencyConfig(BaseModel):
    """Shared claim-store settings for side-effecting tools."""

    model_config = ConfigDict(frozen=True)
    backend: Literal["memory", "sqlite"] = "memory"
    path: str = Field(default=".anycode/tool-idempotency.db", min_length=1)
    redact_sensitive_data: bool = True


class ToolUseContext(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    agent: AgentInfo
    team: TeamInfo | None = None
    cwd: str | None = None
    metadata: dict[str, Any] | None = None
    security_policy: ToolSecurityPolicy | None = None
    idempotency_key: str | None = None
    execution_context: ExecutionContext | None = None


class ToolDefinition(BaseModel):
    """A tool with a Pydantic model for input validation and an async execute function."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    name: str
    description: str
    input_model: type[BaseModel]
    execute: Callable[..., Awaitable[ToolResult]]
    side_effecting: bool = False
    idempotency_key_field: str | None = "idempotency_key"


# -- Agent configuration --


class RetryPolicy(BaseModel):
    """Backoff/retry behavior for transient provider failures."""

    """Tool output with an optional explicit judgment about retry safety."""

    model_config = ConfigDict(frozen=True)
    max_attempts: int = 6
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    jitter: bool = True
    respect_retry_after: bool = True
    call_timeout_seconds: float = 300.0


class ProviderResilienceConfig(BaseModel):
    """Capacity, retry, deadline, and circuit-breaker settings for an LLM provider."""

    model_config = ConfigDict(frozen=True)
    enabled: bool = True
    retry: RetryPolicy = RetryPolicy()
    circuit_failure_threshold: int = 5
    circuit_reset_seconds: float = 120.0
    enable_prompt_cache: bool = True
    max_concurrency: int | None = Field(default=DEFAULT_PROVIDER_CONCURRENCY, ge=1)
    requests_per_minute: int | None = Field(default=None, ge=1)
    capacity_scope: str | None = Field(default=None, min_length=1)
    capacity_wait_timeout_seconds: float | None = Field(default=DEFAULT_PROVIDER_CAPACITY_WAIT_SECONDS, gt=0)


class AgentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    model: str
    provider: str | None = None
    system_prompt: str | None = None
    tools: list[str] | None = None
    max_turns: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    mcp_servers: list[str] | None = None
    context_policy: ContextPolicy | None = None
    verification: tuple[VerificationSensorConfig, ...] = ()
    tool_security: ToolSecurityPolicy | None = None
    provider_resilience: ProviderResilienceConfig | None = None
    execution_context: ExecutionContext | None = None


class AgentState(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: Literal["idle", "running", "completed", "cancelled", "error"] = "idle"
    messages: list[LLMMessage] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    error: str | None = None


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    tool_name: str
    input: dict[str, Any]
    output: str
    duration: float
    retry_safe: bool = True


class AgentRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    success: bool
    output: str
    messages: list[LLMMessage]
    token_usage: TokenUsage
    tool_calls: list[ToolCallRecord]
    handoff_request: HandoffRequest | None = None
    reflections_count: int = 0
    quality_score: float | None = None
    terminal_phase: str | None = None
    stop_reason: StopReason | None = None
    lifecycle_events: list[LifecycleEvent] = []
    context_manifests: list[ContextManifest] = []
    verification_results: list[VerificationResult] = []
    gate_decisions: list[QualityGateDecision] = []
    retries: int = 0


# -- Team --


class TeamConfig(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    name: str
    agents: list[AgentConfig]
    shared_memory: bool | None = None
    max_concurrency: int | None = None
    memory_store: MemoryStore | None = None


class TeamRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    success: bool
    agent_results: dict[str, AgentRunResult]
    total_token_usage: TokenUsage
    handoffs: list[Handoff] | None = None
    route_decisions: list[RouteDecision] | None = None
    cost_report: CostReport | None = None
    lifecycle_events: tuple[LifecycleEvent, ...] = ()
    verification_results: tuple[VerificationResult, ...] = ()
    gate_decisions: tuple[QualityGateDecision, ...] = ()
    stop_reason: StopReason | None = None


# -- Tasks --

TaskStatus = Literal["pending", "in_progress", "completed", "failed", "blocked"]


class Task(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    title: str
    description: str
    status: TaskStatus = "pending"
    assignee: str | None = None
    depends_on: list[str] | None = None
    expected_output: str | None = None
    result: str | None = None
    created_at: datetime
    updated_at: datetime


# -- Orchestrator --


class OrchestratorEvent(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["agent_start", "agent_complete", "task_start", "task_complete", "message", "error"]
    agent: str | None = None
    task: str | None = None
    data: Any = None


class OrchestratorConfig(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    max_concurrency: int | None = None
    default_model: str | None = None
    default_provider: str | None = None
    on_progress: Callable[[OrchestratorEvent], None] | None = None
    memory: MemoryConfig | None = None
    checkpoint: CheckpointConfig | None = None
    approval: ApprovalConfig | None = None
    approval_handler: ApprovalGate | None = None
    mcp_servers: list[MCPServerConfig] | None = None
    handoff_policy: HandoffPolicy | None = None
    max_handoff_depth: int = 3
    routing: RoutingConfig | None = None
    cost: CostConfig | None = None
    reflection: ReflectionConfig | None = None
    rag: RAGConfig | None = None
    verification: tuple[VerificationSensorConfig, ...] = ()
    provider_resilience: ProviderResilienceConfig | None = None
    tool_idempotency: ToolIdempotencyConfig = ToolIdempotencyConfig()


# -- Memory --


class MemoryEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    key: str
    value: str
    metadata: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime | None = None


@runtime_checkable
class MemoryStore(Protocol):
    """Async key-value store interface."""

    async def get(self, key: str) -> MemoryEntry | None: ...
    async def set(self, key: str, value: str, metadata: dict[str, Any] | None = None) -> None: ...
    async def list(self) -> list[MemoryEntry]: ...
    async def delete(self, key: str) -> None: ...
    async def clear(self) -> None: ...


# -- LLM adapter --


ReasoningEffort = Literal["minimal", "low", "medium", "high"]


class LLMChatOptions(BaseModel):
    model_config = ConfigDict(frozen=True)
    model: str
    tools: list[LLMToolDef] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    system_prompt: str | None = None
    enable_prompt_cache: bool = False
    # Reasoning-model controls. `reasoning_effort` maps to OpenAI's field and to
    # a token budget on providers that express thinking as a budget instead.
    reasoning_effort: ReasoningEffort | None = None
    thinking_budget_tokens: int | None = None
    execution_context: ExecutionContext | None = None


class LLMStreamOptions(LLMChatOptions):
    pass


# -- Durable run store --

RunStatus = Literal["running", "paused", "interrupted", "completed", "failed", "cancelled"]
TerminalRunStatus = Literal["completed", "failed", "cancelled"]

TranscriptEventKind = Literal[
    "message",
    "tool_call",
    "tool_result",
    "lifecycle",
    "gate_decision",
    "approval_request",
    "approval_response",
    "compaction",
    "checkpoint",
    "pause",
    "wake",
    "retry",
    "circuit_open",
    "stall_warning",
    "stop",
]


class RunRetentionPolicy(BaseModel):
    """Bounds for pruning terminal durable runs."""

    model_config = ConfigDict(frozen=True)
    max_age_days: float | None = Field(default=None, ge=0)
    max_runs: int | None = Field(default=None, ge=0)
    statuses: tuple[TerminalRunStatus, ...] = ("completed", "failed", "cancelled")


class DurabilityConfig(BaseModel):
    """Opt-in durable persistence for a single agent run."""

    model_config = ConfigDict(frozen=True)
    enabled: bool = False
    run_root: str = ".anycode/runs"
    checkpoint_every_turns: int = 5
    keep_last_checkpoints: int = 3
    heartbeat_seconds: float = 30.0
    redact_sensitive_data: bool = True


WakeKind = Literal["at_time", "on_approval", "on_provider_recovery", "manual"]


class WakeCondition(BaseModel):
    """Why a paused run should resume, persisted so any process can wake it."""

    model_config = ConfigDict(frozen=True)
    kind: WakeKind
    wake_at: datetime | None = None
    approval_id: str | None = None
    note: str = ""


class RunRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    run_id: str
    agent_name: str
    model: str
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    last_heartbeat: datetime
    wake: WakeCondition | None = None
    metadata: dict[str, str] = {}


class TranscriptEvent(BaseModel):
    """One append-only entry in a run's durable event log."""

    model_config = ConfigDict(frozen=True)
    seq: int
    ts: datetime
    kind: TranscriptEventKind
    payload: dict[str, Any] = {}


class BudgetSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    tokens_used: int = 0
    cost_used: float = 0.0
    turns_used: int = 0
    tool_calls_used: int = 0


class GoalCriterion(BaseModel):
    """One machine-checkable done-condition in a goal contract."""

    model_config = ConfigDict(frozen=True)
    id: str
    description: str
    steps: tuple[str, ...] = ()
    passes: bool = False
    evidence: str | None = None


class GoalContract(BaseModel):
    """Machine-readable done-conditions for a multi-session task.

    Criteria may only flip to passing through external verification evidence —
    never through the generating agent's own claim (self-grading is the
    documented failure mode for long-running agents).
    """

    model_config = ConfigDict(frozen=True)
    goal: str
    criteria: tuple[GoalCriterion, ...]

    @property
    def complete(self) -> bool:
        return all(c.passes for c in self.criteria)

    def next_incomplete(self) -> GoalCriterion | None:
        for criterion in self.criteria:
            if not criterion.passes:
                return criterion
        return None

    def mark_passed(self, criterion_id: str, evidence: str) -> GoalContract:
        updated = tuple(c.model_copy(update={"passes": True, "evidence": evidence}) if c.id == criterion_id else c for c in self.criteria)
        return self.model_copy(update={"criteria": updated})


class TurnCheckpoint(BaseModel):
    """Full resumable state of a single agent run at a turn boundary."""

    model_config = ConfigDict(frozen=True)
    run_id: str
    turn: int
    messages: list[LLMMessage]
    token_usage: TokenUsage
    budget: BudgetSnapshot = BudgetSnapshot()
    loop_window: tuple[str, ...] = ()
    last_output: str = ""
    retries: int = 0
    lifecycle_events: list[LifecycleEvent] = []
    context_manifests: list[ContextManifest] = []
    verification_results: list[VerificationResult] = []
    gate_decisions: list[QualityGateDecision] = []
    created_at: datetime


@runtime_checkable
class LLMAdapter(Protocol):
    """Provider-agnostic LLM interface."""

    @property
    def name(self) -> str: ...
    async def chat(self, messages: list[LLMMessage], options: LLMChatOptions) -> LLMResponse: ...
    def stream(self, messages: list[LLMMessage], options: LLMStreamOptions) -> AsyncIterable[StreamEvent]: ...


# -- Runner types --


class RunnerStreamingConfig(BaseModel):
    """Controls whether a turn consumes the provider stream incrementally.

    When enabled, the runner emits ``text``/``thinking`` events as the provider
    produces them and assembles the same final ``LLMResponse`` the non-streaming
    path returns. ``fallback_to_chat`` retries the turn via ``chat()`` if the
    stream fails before any output or tool side effect is emitted.
    """

    model_config = ConfigDict(frozen=True)
    enabled: bool = True
    fallback_to_chat: bool = True


class RunnerOptions(BaseModel):
    model_config = ConfigDict(frozen=True)
    model: str
    system_prompt: str | None = None
    max_turns: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    allowed_tools: list[str] | None = None
    agent_name: str | None = None
    agent_role: str | None = None
    verification: tuple[VerificationSensorConfig, ...] = ()
    reasoning_effort: ReasoningEffort | None = None
    thinking_budget_tokens: int | None = None
    streaming: RunnerStreamingConfig | None = None
    tool_security: ToolSecurityPolicy | None = None
    execution_context: ExecutionContext | None = None


class RunResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    messages: list[LLMMessage]
    output: str
    tool_calls: list[ToolCallRecord]
    token_usage: TokenUsage
    turns: int
    handoff_request: HandoffRequest | None = None
    terminal_phase: str | None = None
    stop_reason: StopReason | None = None
    lifecycle_events: list[LifecycleEvent] = []
    context_manifests: list[ContextManifest] = []
    verification_results: list[VerificationResult] = []
    gate_decisions: list[QualityGateDecision] = []
    retries: int = 0


class PoolStatus(BaseModel):
    model_config = ConfigDict(frozen=True)
    total: int
    idle: int
    running: int
    completed: int
    cancelled: int = 0
    error: int


SchedulingStrategy = Literal["round-robin", "least-busy", "capability-match", "dependency-first"]


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    from_agent: str
    to_agent: str
    content: str
    timestamp: datetime


TaskQueueEvent = Literal["task:ready", "task:complete", "task:failed", "all:complete"]


class BatchToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    name: str
    input: dict[str, Any]


# -- Telemetry --


class TraceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool = False
    service_name: str = "anycode"
    exporter: Literal["otlp", "console", "jsonl", "none"] = "console"
    endpoint: str | None = None
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    redact_sensitive_data: bool = True
    max_recorded_spans: int = Field(default=10_000, ge=1)
    max_recorded_events: int = Field(default=10_000, ge=1)
    max_metric_series: int = Field(default=1_000, ge=1)
    max_histogram_samples: int = Field(default=1_000, ge=1)
    capture_profile: Literal["off", "metadata", "redacted", "full"] = "redacted"
    max_attribute_length: int = Field(default=4_096, ge=16)
    max_attribute_count: int = Field(default=128, ge=1)
    telemetry_buffer_capacity: int = Field(default=1_000, ge=1)


class SpanAttributes(BaseModel):
    model_config = ConfigDict(frozen=True)
    run_id: str | None = None
    agent_name: str | None = None
    tool_name: str | None = None
    task_id: str | None = None
    model: str | None = None
    provider: str | None = None
    token_input: int = 0
    token_output: int = 0
    cost_usd: float = 0.0
    turn_number: int = 0
    phase: str | None = None
    stop_reason: str | None = None
    recoverable: bool | None = None
    retry_count: int = 0
    principal: str | None = None
    tenant_scope: str | None = None
    classification: str | None = None
    region: str | None = None


# -- Guardrails --


class GuardrailConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_tokens_per_agent: int | None = None
    max_tokens_per_team: int | None = None
    max_cost_usd: float | None = None
    max_turns: int | None = None
    max_tool_calls: int | None = None
    blocked_tools: list[str] | None = None
    require_approval_tools: list[str] | None = None
    output_validators: list[str] | None = None


class BudgetStatus(BaseModel):
    model_config = ConfigDict(frozen=True)
    tokens_used: int = 0
    tokens_limit: int | None = None
    cost_used: float = 0.0
    cost_limit: float | None = None
    turns_used: int = 0
    turns_limit: int | None = None
    tool_calls_used: int = 0
    tool_calls_limit: int | None = None
    exhausted: bool = False


class ValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    valid: bool
    reason: str | None = None
    retry: bool = False


@runtime_checkable
class OutputValidator(Protocol):
    async def validate(self, output: str, context: AgentInfo) -> ValidationResult: ...


@runtime_checkable
class TurnHook(Protocol):
    async def before_turn(self, messages: list[LLMMessage], context: AgentInfo) -> list[LLMMessage]: ...
    async def after_turn(self, response: LLMResponse, context: AgentInfo) -> LLMResponse: ...


# -- Structured output --


class StructuredOutputConfig(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    schema_class: type[BaseModel]
    max_retries: int = 2


class StructuredRunResult[T](BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    parsed: Any  # T — validated Pydantic instance
    raw_output: str
    messages: list[LLMMessage]
    token_usage: TokenUsage
    tool_calls: list[ToolCallRecord]
    turns: int


class StructuredAgentResult[T](BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    success: bool
    parsed: Any  # T | None — validated Pydantic instance or None
    output: str
    messages: list[LLMMessage]
    token_usage: TokenUsage
    tool_calls: list[ToolCallRecord]


# -- Vector store --


class VectorSearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    text: str
    score: float
    metadata: dict[str, Any] | None = None


@runtime_checkable
class VectorStore(Protocol):
    """Async semantic similarity search interface."""

    async def add(self, texts: list[str], metadata: list[dict[str, Any]] | None = None) -> list[str]: ...
    async def search(self, query: str, top_k: int = 5) -> list[VectorSearchResult]: ...
    async def delete(self, ids: list[str]) -> None: ...
    async def clear(self) -> None: ...


class MemoryConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    backend: Literal["memory", "sqlite", "redis"] = "memory"
    path: str | None = None
    url: str | None = None
    vector_backend: Literal["none", "memory", "chromadb"] = "none"
    vector_path: str | None = None
    redact_sensitive_data: bool = True


# -- Checkpoint --


class CheckpointConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool = False
    backend: Literal["filesystem", "sqlite"] = "filesystem"
    path: str = ".anycode/checkpoints"
    keep_last: int = 5
    redact_sensitive_data: bool = True


class CheckpointData(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    workflow_id: str
    version: int = CHECKPOINT_FORMAT_VERSION
    tasks: list[Task]
    agent_results: dict[str, AgentRunResult]
    wave_index: int
    total_token_usage: TokenUsage
    created_at: datetime
    metadata: dict[str, Any] | None = None


@runtime_checkable
class CheckpointStore(Protocol):
    """Async checkpoint persistence interface."""

    async def save(self, checkpoint: CheckpointData) -> None: ...
    async def load(self, checkpoint_id: str) -> CheckpointData | None: ...
    async def latest(self, workflow_id: str) -> CheckpointData | None: ...
    async def list_checkpoints(self, workflow_id: str) -> list[str]: ...
    async def delete(self, checkpoint_id: str) -> None: ...
    async def prune(self, workflow_id: str, keep_last: int) -> None: ...


# -- Human-in-the-loop --


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    type: Literal["task", "tool_call", "output"]
    agent: str
    description: str
    context: dict[str, Any] | None = None
    created_at: datetime


class ApprovalResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    request_id: str
    approved: bool
    reason: str | None = None
    modified_input: dict[str, Any] | None = None
    responded_at: datetime


@runtime_checkable
class ApprovalGate(Protocol):
    """Async approval gate interface."""

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse: ...


class ApprovalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool = False
    timeout_seconds: float = 300.0
    default_on_timeout: Literal["approve", "reject"] = "reject"
    require_approval_tools: list[str] | None = None
    require_approval_tasks: bool = False


# -- MCP (Model Context Protocol) --


class MCPTrustPolicy(BaseModel):
    """Trust boundary for MCP subprocess and HTTP transports."""

    model_config = ConfigDict(frozen=True)
    allow_stdio: bool = True
    allow_insecure_http: bool = False
    allow_private_networks: bool = False
    allowed_hosts: tuple[str, ...] = ()


class MCPServerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    transport: Literal["stdio", "sse", "streamable-http"]
    command: str | None = None
    args: list[str] | None = None
    url: str | None = None
    env: dict[str, str] | None = None
    timeout: float = 30.0
    # HTTP auth: static headers plus an optional env var whose value is sent as a
    # Bearer token. Secrets are resolved here, never surfaced into model context.
    headers: dict[str, str] | None = None
    auth_token_env: str | None = None
    trust: MCPTrustPolicy = MCPTrustPolicy()


class MCPToolInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    server: str
    name: str
    description: str
    input_schema: dict[str, Any]
    side_effecting: bool = True


# -- Agent handoff --


class HandoffRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    to_agent: str
    summary: str
    reason: str


class Handoff(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    from_agent: str
    to_agent: str
    context: list[LLMMessage]
    summary: str
    reason: str
    metadata: dict[str, Any] | None = None
    created_at: datetime


@runtime_checkable
class HandoffPolicy(Protocol):
    """Evaluates whether an agent should hand off to another agent."""

    async def should_handoff(self, agent: AgentInfo, result: RunResult) -> HandoffRequest | None: ...


# -- Intelligent routing --

ComplexityLevel = Literal["trivial", "simple", "moderate", "complex", "expert"]


class RoutingRule(BaseModel):
    model_config = ConfigDict(frozen=True)
    condition: str
    target_model: str
    target_provider: str | None = None
    priority: int = 0


class RoutingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool = False
    rules: list[RoutingRule] | None = None
    default_model: str | None = None
    default_provider: str | None = None
    classify_with_llm: bool = False


class RouteDecision(BaseModel):
    model_config = ConfigDict(frozen=True)
    task_id: str
    original_model: str
    routed_model: str
    routed_provider: str | None = None
    complexity: ComplexityLevel
    reason: str


@runtime_checkable
class Router(Protocol):
    """Route tasks to optimal models based on complexity."""

    async def route(self, task: Task, agents: list[AgentConfig]) -> RouteDecision | None: ...


# -- Cost engine --


class ModelPricing(BaseModel):
    model_config = ConfigDict(frozen=True)
    model: str
    provider: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    cached_input_cost_per_1k: float | None = None


class CostConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool = True
    budget_usd: float | None = None
    alert_threshold: float = 0.8
    on_budget_exceeded: Literal["stop", "warn", "continue"] = "stop"
    custom_pricing: list[ModelPricing] | None = None


class CostBreakdown(BaseModel):
    model_config = ConfigDict(frozen=True)
    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    input_cost_usd: float
    output_cost_usd: float
    cache_read_cost_usd: float = 0.0
    total_cost_usd: float
    calls: int


class CostReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    total_cache_creation_input_tokens: int = 0
    total_cache_read_input_tokens: int = 0
    by_agent: list[CostBreakdown]
    by_model: list[CostBreakdown]
    budget_usd: float | None = None
    budget_remaining_usd: float | None = None


# -- Reflection --


class CriticResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    score: float
    passed: bool
    feedback: str
    suggestions: list[str] = []


@runtime_checkable
class Critic(Protocol):
    """Evaluates the quality of an agent's output."""

    async def evaluate(self, output: str, prompt: str, context: AgentInfo) -> CriticResult: ...


class ReflectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    enabled: bool = False
    mode: Literal["self", "peer", "custom"] = "self"
    critic_model: str | None = None
    critic_provider: str | None = None
    quality_threshold: float = 0.7
    max_reflections: int = 2
    critic_prompt: str | None = None
    custom_critic: Critic | None = None


# -- RAG memory --


class RAGEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    text: str
    source: str
    score: float
    timestamp: datetime


class RAGContext(BaseModel):
    model_config = ConfigDict(frozen=True)
    entries: list[RAGEntry]
    total_tokens: int


class RAGConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool = False
    auto_index: bool = True
    top_k: int = 5
    min_relevance: float = 0.3
    max_context_tokens: int | None = 2000
    index_tool_results: bool = True
    namespace: str = "default"


# -- Harness component registry --

HarnessComponentKind = Literal[
    "prompt",
    "tool",
    "context_policy",
    "routing_policy",
    "verification",
    "memory",
    "provider",
]

HarnessComponentOwner = Literal["core", "config", "user", "plugin"]


class HarnessComponent(BaseModel):
    """Single editable or inspectable artifact of the agent harness."""

    model_config = ConfigDict(frozen=True)
    id: str
    kind: HarnessComponentKind
    source: str
    editable: bool
    owner: HarnessComponentOwner
    checksum: str
    description: str
    metadata: dict[str, str] = Field(default_factory=dict)


class HarnessManifest(BaseModel):
    """Snapshot of every harness component for a configured run."""

    model_config = ConfigDict(frozen=True)
    manifest_version: str = "1"
    components: tuple[HarnessComponent, ...]
    created_at: datetime
    checksum: str
    notes: str | None = None


# -- Trajectory evidence corpus --


class FailureCategory(StrEnum):
    TOOL_ARGUMENT_ERROR = "tool_argument_error"
    TOOL_RUNTIME_ERROR = "tool_runtime_error"
    CONTEXT_LOSS = "context_loss"
    EARLY_STOPPING = "early_stopping"
    VERIFICATION_FAILURE = "verification_failure"
    BUDGET_EXCEEDED = "budget_exceeded"
    POLICY_BLOCKED = "policy_blocked"
    SECURITY_ANOMALY = "security_anomaly"
    SUCCESS = "success"
    UNKNOWN = "unknown"


EvidenceSeverity = Literal["low", "medium", "high", "critical"]


class EvidencePacket(BaseModel):
    """A compact failure (or success) artifact tied to raw trace event IDs."""

    model_config = ConfigDict(frozen=True)
    id: str
    category: FailureCategory
    summary: str
    event_ids: tuple[str, ...] = ()
    severity: EvidenceSeverity = "medium"
    suggested_component_ids: tuple[str, ...] = ()
    evidence: dict[str, str] = Field(default_factory=dict)


class TrajectoryEvent(BaseModel):
    """One normalized event in a run's trajectory (lifecycle / tool / model)."""

    model_config = ConfigDict(frozen=True)
    id: str
    kind: Literal["lifecycle", "tool_call", "model_turn", "verification", "error"]
    name: str
    timestamp: float
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)


class RunSummary(BaseModel):
    """Top-level outcome summary for a single run."""

    model_config = ConfigDict(frozen=True)
    run_id: str
    task: str
    outcome: Literal["pass", "fail", "error"]
    stop_reason: str | None = None
    cost_usd: float = 0.0
    runtime_seconds: float = 0.0
    turns: int = 0
    quality_gate: Literal["pass", "warn", "fail", "unknown"] = "unknown"
    verification_failures: int = 0


class FailureMapEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    category: FailureCategory
    count: int
    representative_event_ids: tuple[str, ...] = ()


class TrajectoryEvidence(BaseModel):
    """All distilled artifacts for one run."""

    model_config = ConfigDict(frozen=True)
    run_summary: RunSummary
    failure_map: tuple[FailureMapEntry, ...] = ()
    decision_timeline: tuple[TrajectoryEvent, ...] = ()
    evidence_packets: tuple[EvidencePacket, ...] = ()
    raw_trace_path: str | None = None
    manifest_checksum: str | None = None


# -- Controlled evolution loop --


class HarnessChangePrediction(BaseModel):
    """Falsifiable forecast about a candidate harness change."""

    model_config = ConfigDict(frozen=True)
    metric: str
    expected_direction: Literal["increase", "decrease", "unchanged"]
    expected_delta: float | None = None
    rationale: str


HarnessChangeStatus = Literal[
    "proposed",
    "applied",
    "evaluated",
    "accepted",
    "rejected",
    "rolled_back",
]


class HarnessChangeEdit(BaseModel):
    """One concrete edit to a single component, captured as a reviewable diff."""

    model_config = ConfigDict(frozen=True)
    component_id: str
    before_checksum: str
    after_checksum: str
    diff: str
    note: str = ""


class HarnessChangeManifest(BaseModel):
    """Predicted, falsifiable change to one or more harness components."""

    model_config = ConfigDict(frozen=True)
    id: str
    component_ids: tuple[str, ...]
    evidence_packet_ids: tuple[str, ...] = ()
    summary: str
    predictions: tuple[HarnessChangePrediction, ...]
    rollback_plan: str
    safety_review_required: bool = True
    edits: tuple[HarnessChangeEdit, ...] = ()
    created_at: datetime | None = None


class AcceptanceThresholds(BaseModel):
    """Hard policy floors that no blueprint optimization may relax."""

    model_config = ConfigDict(frozen=True)
    min_pass_delta: int = 0
    max_regressions: int = 0
    max_runtime_delta_seconds: float | None = None
    max_cost_delta_usd: float | None = None
    block_on_safety_regression: bool = True


class HarnessChangeOutcome(BaseModel):
    """Result of measuring a candidate manifest against baseline + thresholds."""

    model_config = ConfigDict(frozen=True)
    manifest_id: str
    status: HarnessChangeStatus
    baseline_passed: int
    candidate_passed: int
    regressions: tuple[str, ...] = ()
    improvements: tuple[str, ...] = ()
    predicted_vs_measured: tuple[dict[str, str | float | bool | None], ...] = ()
    reasons: tuple[str, ...] = ()
    patch_path: str | None = None


# -- Meta-harness optimization --


class EvolutionBlueprint(BaseModel):
    """Versioned blueprint that describes the evolution loop itself."""

    model_config = ConfigDict(frozen=True)
    id: str
    worker_seed: str
    evaluator_prompt_id: str
    evolution_prompt_id: str
    evidence_policy_id: str
    max_iterations: int = 3
    acceptance_policy_id: str = "default"
    description: str = ""
    safety_floors: AcceptanceThresholds = Field(default_factory=AcceptanceThresholds)


class MetaHarnessReport(BaseModel):
    """Summary of running one or more blueprints across train + held-out suites."""

    model_config = ConfigDict(frozen=True)
    blueprint_id: str
    train_scores: tuple[float, ...] = ()
    heldout_scores: tuple[float, ...] = ()
    convergence_iterations: tuple[int, ...] = ()
    accepted_changes: int = 0
    rejected_changes: int = 0
    total_cost_usd: float = 0.0
    regression_rate: float = 0.0
    notes: str = ""


# -- Plugin / extension ecosystem --

PluginSource = Literal["manual", "entry_point", "builtin"]


class PluginTrustPolicy(BaseModel):
    """Allowlist applied before third-party entry points are imported."""

    model_config = ConfigDict(frozen=True)
    allowed_entry_points: tuple[str, ...] = ()
    allowed_distributions: tuple[str, ...] = ()
    allow_unlisted: bool = False


class PluginManifest(BaseModel):
    """Static metadata for a plugin — name, version, and a short human description."""

    model_config = ConfigDict(frozen=True)
    name: str
    version: str = "0.0.0"
    description: str = ""
    homepage: str | None = None
    source: PluginSource = "manual"


ProviderFactory = Callable[..., Awaitable["LLMAdapter"]]


@runtime_checkable
class Plugin(Protocol):
    """A single extension bundle wiring tools, providers, sensors, and hooks into AnyCode.

    Every accessor is optional — return an empty sequence/mapping when the plugin does not
    contribute that kind of extension. The manifest is required and is the only piece used
    for inspection and duplicate detection.
    """

    @property
    def manifest(self) -> PluginManifest: ...

    def tools(self) -> Sequence[ToolDefinition]: ...

    def provider_factories(self) -> Mapping[str, ProviderFactory]: ...

    def sensors(self) -> Sequence[VerificationSensorConfig]: ...

    def turn_hooks(self) -> Sequence[TurnHook]: ...


class PluginInstallation(BaseModel):
    """Records what a plugin contributed when installed into a `PluginRegistry`."""

    model_config = ConfigDict(frozen=True)
    manifest: PluginManifest
    tool_names: tuple[str, ...] = ()
    provider_names: tuple[str, ...] = ()
    sensor_names: tuple[str, ...] = ()
    turn_hook_count: int = 0
