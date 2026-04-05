"""AnyCode — scalable multi-agent AI orchestration framework for Python."""

from anycode.checkpoint.manager import CheckpointManager
from anycode.checkpoint.store import FilesystemCheckpointStore
from anycode.collaboration.kv_store import InMemoryStore
from anycode.collaboration.message_bus import MessageBus
from anycode.collaboration.shared_mem import SharedMemory
from anycode.collaboration.team import Team
from anycode.constants import (
    ORCH_EVENT_AGENT_COMPLETE,
    ORCH_EVENT_AGENT_START,
    ORCH_EVENT_BROADCAST,
    ORCH_EVENT_ERROR,
    ORCH_EVENT_MESSAGE,
    ORCH_EVENT_TASK_COMPLETE,
    ORCH_EVENT_TASK_START,
    QUEUE_EVENT_ALL_COMPLETE,
    QUEUE_EVENT_TASK_COMPLETE,
    QUEUE_EVENT_TASK_FAILED,
    QUEUE_EVENT_TASK_READY,
)
from anycode.core.agent import Agent
from anycode.core.orchestrator import AnyCode, TaskSpec
from anycode.core.pool import AgentPool
from anycode.core.runner import AgentRunner
from anycode.core.scheduler import Scheduler
from anycode.guardrails.budget import BudgetTracker, estimate_cost
from anycode.guardrails.hooks import HookRunner, LoggingHook
from anycode.guardrails.validators import (
    BlocklistValidator,
    ContainsValidator,
    MaxLengthValidator,
    run_validators,
)
from anycode.helpers.concurrency_gate import Semaphore
from anycode.helpers.usage_tracker import EMPTY_USAGE, merge_usage
from anycode.helpers.uuid7 import uuid7
from anycode.hitl.approval import ApprovalManager
from anycode.hitl.channels import CallbackApprovalGate, StdinApprovalGate, WebhookApprovalGate
from anycode.memory.composite import CompositeMemory
from anycode.memory.factory import create_memory_store
from anycode.memory.vector_store import InMemoryVectorStore
from anycode.providers.adapter import create_adapter
from anycode.structured.output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    build_retry_prompt,
    parse_structured_output,
    schema_to_openai_response_format,
    schema_to_tool_def,
)
from anycode.tasks.queue import TaskQueue
from anycode.tasks.task import create_task, get_task_dependency_order, is_task_ready, validate_task_dependencies
from anycode.telemetry.events import EventEmitter, TelemetryEvent
from anycode.telemetry.metrics import MetricsCollector, Timer
from anycode.telemetry.tracer import ConsoleExporter, Span, Tracer
from anycode.tools.built_in import BUILT_IN_TOOLS, register_built_in_tools
from anycode.tools.executor import ToolExecutor
from anycode.tools.registry import ToolRegistry, define_tool
from anycode.types import (
    AgentConfig,
    AgentInfo,
    AgentRunResult,
    AgentState,
    ApprovalConfig,
    ApprovalGate,
    ApprovalRequest,
    ApprovalResponse,
    BudgetStatus,
    CheckpointConfig,
    CheckpointData,
    CheckpointStore,
    ContentBlock,
    GuardrailConfig,
    ImageBlock,
    LLMAdapter,
    LLMChatOptions,
    LLMMessage,
    LLMResponse,
    LLMStreamOptions,
    LLMToolDef,
    MemoryConfig,
    MemoryEntry,
    MemoryStore,
    OrchestratorConfig,
    OrchestratorEvent,
    OutputValidator,
    PoolStatus,
    RunnerOptions,
    RunResult,
    SchedulingStrategy,
    SpanAttributes,
    StreamEvent,
    StructuredAgentResult,
    StructuredOutputConfig,
    StructuredRunResult,
    Task,
    TaskStatus,
    TeamConfig,
    TeamRunResult,
    TextBlock,
    TokenUsage,
    ToolCallRecord,
    ToolDefinition,
    ToolResult,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseContext,
    TraceConfig,
    TurnHook,
    ValidationResult,
    VectorSearchResult,
    VectorStore,
)

__all__ = [
    # Constants — event names
    "QUEUE_EVENT_TASK_READY",
    "QUEUE_EVENT_TASK_COMPLETE",
    "QUEUE_EVENT_TASK_FAILED",
    "QUEUE_EVENT_ALL_COMPLETE",
    "ORCH_EVENT_AGENT_START",
    "ORCH_EVENT_AGENT_COMPLETE",
    "ORCH_EVENT_TASK_START",
    "ORCH_EVENT_TASK_COMPLETE",
    "ORCH_EVENT_MESSAGE",
    "ORCH_EVENT_ERROR",
    "ORCH_EVENT_BROADCAST",
    # Core
    "AnyCode",
    "Agent",
    "AgentRunner",
    "AgentPool",
    "Scheduler",
    "TaskSpec",
    # Providers
    "create_adapter",
    # Collaboration
    "Team",
    "MessageBus",
    "SharedMemory",
    "InMemoryStore",
    # Memory
    "CompositeMemory",
    "InMemoryVectorStore",
    "create_memory_store",
    # Checkpoint
    "CheckpointManager",
    "FilesystemCheckpointStore",
    # HITL
    "ApprovalManager",
    "CallbackApprovalGate",
    "StdinApprovalGate",
    "WebhookApprovalGate",
    # Tasks
    "TaskQueue",
    "create_task",
    "is_task_ready",
    "get_task_dependency_order",
    "validate_task_dependencies",
    # Tools
    "ToolRegistry",
    "ToolExecutor",
    "define_tool",
    "BUILT_IN_TOOLS",
    "register_built_in_tools",
    # Helpers
    "Semaphore",
    "EMPTY_USAGE",
    "merge_usage",
    "uuid7",
    # Telemetry
    "Tracer",
    "Span",
    "ConsoleExporter",
    "MetricsCollector",
    "Timer",
    "EventEmitter",
    "TelemetryEvent",
    # Guardrails
    "BudgetTracker",
    "estimate_cost",
    "HookRunner",
    "LoggingHook",
    "run_validators",
    "MaxLengthValidator",
    "ContainsValidator",
    "BlocklistValidator",
    # Structured output
    "STRUCTURED_OUTPUT_TOOL_NAME",
    "schema_to_tool_def",
    "schema_to_openai_response_format",
    "parse_structured_output",
    "build_retry_prompt",
    # Types
    "ContentBlock",
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ImageBlock",
    "LLMMessage",
    "TokenUsage",
    "LLMResponse",
    "StreamEvent",
    "LLMToolDef",
    "ToolUseContext",
    "AgentInfo",
    "ToolResult",
    "ToolDefinition",
    "AgentConfig",
    "AgentState",
    "ToolCallRecord",
    "AgentRunResult",
    "TeamConfig",
    "TeamRunResult",
    "TaskStatus",
    "Task",
    "OrchestratorEvent",
    "OrchestratorConfig",
    "MemoryEntry",
    "MemoryStore",
    "MemoryConfig",
    "VectorStore",
    "VectorSearchResult",
    "CheckpointConfig",
    "CheckpointData",
    "CheckpointStore",
    "ApprovalRequest",
    "ApprovalResponse",
    "ApprovalGate",
    "ApprovalConfig",
    "LLMChatOptions",
    "LLMStreamOptions",
    "LLMAdapter",
    "RunnerOptions",
    "RunResult",
    "PoolStatus",
    "SchedulingStrategy",
    "TraceConfig",
    "SpanAttributes",
    "GuardrailConfig",
    "BudgetStatus",
    "ValidationResult",
    "OutputValidator",
    "TurnHook",
    "StructuredOutputConfig",
    "StructuredRunResult",
    "StructuredAgentResult",
]
