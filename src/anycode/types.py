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
