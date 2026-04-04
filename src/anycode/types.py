"""AnyCode — Pydantic models for the entire type surface."""

from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

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


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock | ImageBlock


# -- Conversation messages --


class LLMMessage(BaseModel):
    model_config = ConfigDict(frozen=True)
    role: Literal["user", "assistant"]
    content: list[ContentBlock]


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)
    input_tokens: int = 0
    output_tokens: int = 0


class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    content: list[ContentBlock]
    model: str
    stop_reason: str
    usage: TokenUsage


class StreamEvent(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["text", "tool_use", "tool_result", "done", "error"]
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


class AgentInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    role: str
    model: str


class TeamInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    agents: list[str]


class ToolUseContext(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    agent: AgentInfo
    team: TeamInfo | None = None
    cwd: str | None = None
    metadata: dict[str, Any] | None = None


class ToolDefinition(BaseModel):
    """A tool with a Pydantic model for input validation and an async execute function."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    name: str
    description: str
    input_model: type[BaseModel]
    execute: Callable[..., Awaitable[ToolResult]]


# -- Agent configuration --


class AgentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    model: str
    provider: Literal["anthropic", "openai"] | None = None
    system_prompt: str | None = None
    tools: list[str] | None = None
    max_turns: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None


class AgentState(BaseModel):
    status: Literal["idle", "running", "completed", "error"] = "idle"
    messages: list[LLMMessage] = []
    token_usage: TokenUsage = TokenUsage()
    error: str | None = None


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    tool_name: str
    input: dict[str, Any]
    output: str
    duration: float


class AgentRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    success: bool
    output: str
    messages: list[LLMMessage]
    token_usage: TokenUsage
    tool_calls: list[ToolCallRecord]


# -- Team --


class TeamConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    agents: list[AgentConfig]
    shared_memory: bool | None = None
    max_concurrency: int | None = None


class TeamRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    success: bool
    agent_results: dict[str, AgentRunResult]
    total_token_usage: TokenUsage


# -- Tasks --

TaskStatus = Literal["pending", "in_progress", "completed", "failed", "blocked"]


class Task(BaseModel):
    id: str
    title: str
    description: str
    status: TaskStatus = "pending"
    assignee: str | None = None
    depends_on: list[str] | None = None
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
    default_provider: Literal["anthropic", "openai"] | None = None
    on_progress: Callable[[OrchestratorEvent], None] | None = None


# -- Memory --


class MemoryEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    key: str
    value: str
    metadata: dict[str, Any] | None = None
    created_at: datetime


@runtime_checkable
class MemoryStore(Protocol):
    """Async key-value store interface."""

    async def get(self, key: str) -> MemoryEntry | None: ...
    async def set(self, key: str, value: str, metadata: dict[str, Any] | None = None) -> None: ...
    async def list(self) -> list[MemoryEntry]: ...
    async def delete(self, key: str) -> None: ...
    async def clear(self) -> None: ...


# -- LLM adapter --


class LLMChatOptions(BaseModel):
    model_config = ConfigDict(frozen=True)
    model: str
    tools: list[LLMToolDef] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    system_prompt: str | None = None


class LLMStreamOptions(LLMChatOptions):
    pass


@runtime_checkable
class LLMAdapter(Protocol):
    """Provider-agnostic LLM interface."""

    @property
    def name(self) -> str: ...
    async def chat(self, messages: list[LLMMessage], options: LLMChatOptions) -> LLMResponse: ...
    def stream(self, messages: list[LLMMessage], options: LLMStreamOptions) -> AsyncIterable[StreamEvent]: ...


# -- Runner types --


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


class RunResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    messages: list[LLMMessage]
    output: str
    tool_calls: list[ToolCallRecord]
    token_usage: TokenUsage
    turns: int


class PoolStatus(BaseModel):
    model_config = ConfigDict(frozen=True)
    total: int
    idle: int
    running: int
    completed: int
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
    exporter: Literal["otlp", "console", "none"] = "console"
    endpoint: str | None = None
    sample_rate: float = 1.0


class SpanAttributes(BaseModel):
    model_config = ConfigDict(frozen=True)
    agent_name: str | None = None
    tool_name: str | None = None
    task_id: str | None = None
    model: str | None = None
    provider: str | None = None
    token_input: int = 0
    token_output: int = 0
    cost_usd: float = 0.0
    turn_number: int = 0


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
