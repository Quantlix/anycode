"""Centralized constants for the AnyCode framework.

All magic numbers, event names, and shared configuration defaults
are defined here to provide a single source of truth.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Time & unit conversions
# ---------------------------------------------------------------------------

MS_PER_SECOND = 1000

# ---------------------------------------------------------------------------
# Task queue event names (colon-notation, internal bus)
# ---------------------------------------------------------------------------

QUEUE_EVENT_TASK_READY = "task:ready"
QUEUE_EVENT_TASK_COMPLETE = "task:complete"
QUEUE_EVENT_TASK_FAILED = "task:failed"
QUEUE_EVENT_ALL_COMPLETE = "all:complete"

# ---------------------------------------------------------------------------
# Orchestrator event types (underscore-notation, external consumers)
# ---------------------------------------------------------------------------

ORCH_EVENT_AGENT_START = "agent_start"
ORCH_EVENT_AGENT_COMPLETE = "agent_complete"
ORCH_EVENT_TASK_START = "task_start"
ORCH_EVENT_TASK_COMPLETE = "task_complete"
ORCH_EVENT_MESSAGE = "message"
ORCH_EVENT_ERROR = "error"
ORCH_EVENT_BROADCAST = "broadcast"

# ---------------------------------------------------------------------------
# Telemetry lifecycle event names (dot-notation)
# ---------------------------------------------------------------------------

TEL_EVENT_AGENT_START = "agent.start"
TEL_EVENT_AGENT_COMPLETE = "agent.complete"
TEL_EVENT_AGENT_ERROR = "agent.error"
TEL_EVENT_TURN_START = "turn.start"
TEL_EVENT_TURN_COMPLETE = "turn.complete"
TEL_EVENT_TOOL_START = "tool.start"
TEL_EVENT_TOOL_COMPLETE = "tool.complete"
TEL_EVENT_LLM_CALL_START = "llm.call.start"
TEL_EVENT_LLM_CALL_COMPLETE = "llm.call.complete"
TEL_EVENT_BUDGET_WARNING = "budget.warning"
TEL_EVENT_BUDGET_EXHAUSTED = "budget.exhausted"

# ---------------------------------------------------------------------------
# Content block type identifiers
# ---------------------------------------------------------------------------

BLOCK_TYPE_TEXT = "text"
BLOCK_TYPE_TOOL_USE = "tool_use"
BLOCK_TYPE_TOOL_RESULT = "tool_result"
BLOCK_TYPE_IMAGE = "image"
BLOCK_TYPE_BASE64 = "base64"

# ---------------------------------------------------------------------------
# LLM stop reasons
# ---------------------------------------------------------------------------

STOP_REASON_END_TURN = "end_turn"
STOP_REASON_TOOL_USE = "tool_use"
STOP_REASON_MAX_TOKENS = "max_tokens"
STOP_REASON_CONTENT_FILTER = "content_filter"

# ---------------------------------------------------------------------------
# LLM & concurrency defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_CONCURRENCY = 5
DEFAULT_TOOL_CONCURRENCY = 4
DEFAULT_TURN_LIMIT = 10
DEFAULT_MAX_TOKENS = 4096
MAX_VALIDATION_RETRIES = 3

# ---------------------------------------------------------------------------
# Provider identifiers
# ---------------------------------------------------------------------------

PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"
PROVIDER_GOOGLE = "google"
PROVIDER_OLLAMA = "ollama"
PROVIDER_BEDROCK = "bedrock"
PROVIDER_AZURE = "azure"

# ---------------------------------------------------------------------------
# String truncation limits
# ---------------------------------------------------------------------------

AGENT_ROLE_MAX_LENGTH = 50
TOOL_CONTEXT_ROLE_MAX_LENGTH = 60
COORDINATOR_ROLE_PREVIEW_LENGTH = 80
DEPENDENCY_CONTEXT_MAX_LENGTH = 500
MEMORY_DISPLAY_MAX_LENGTH = 180
MEMORY_DISPLAY_TRUNCATE_AT = 177

# ---------------------------------------------------------------------------
# Bash tool exit codes
# ---------------------------------------------------------------------------

EXIT_CODE_TIMEOUT = 124
EXIT_CODE_NOT_FOUND = 127

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

CHECKPOINT_FORMAT_VERSION = 1

# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

DEFAULT_ENCODING = "utf-8"

# ---------------------------------------------------------------------------
# HITL / review UI
# ---------------------------------------------------------------------------

APPROVAL_BOX_WIDTH = 55
APPROVAL_CONTEXT_MAX_DISPLAY = 80
APPROVAL_CONTEXT_TRUNCATE_AT = 77
WEBHOOK_TIMEOUT_S = 30
POLL_INTERVAL_S = 5.0

BOX_TOP_LEFT = "\u250c"
BOX_TOP_RIGHT = "\u2510"
BOX_BOTTOM_LEFT = "\u2514"
BOX_BOTTOM_RIGHT = "\u2518"
BOX_HORIZONTAL = "\u2500"
BOX_VERTICAL = "\u2502"

# ---------------------------------------------------------------------------
# Memory / storage defaults
# ---------------------------------------------------------------------------

REDIS_KEY_PREFIX = "anycode:mem:"
REDIS_MAX_RETRIES = 3
REDIS_RETRY_BASE_SECONDS = 0.5
CHROMADB_DEFAULT_COLLECTION = "anycode_memory"
CHROMADB_DEFAULT_PORT = 8000

# ---------------------------------------------------------------------------
# Search / grep
# ---------------------------------------------------------------------------

GREP_MATCH_CEILING = 100
GREP_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".svn",
        ".hg",
        "node_modules",
        ".next",
        "dist",
        "build",
        "__pycache__",
        ".venv",
    }
)

# ---------------------------------------------------------------------------
# Bash tool defaults
# ---------------------------------------------------------------------------

BASH_TIMEOUT_LIMIT_S = 30

# ---------------------------------------------------------------------------
# Provider defaults
# ---------------------------------------------------------------------------

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
OLLAMA_REQUEST_TIMEOUT_S = 120.0
AZURE_DEFAULT_API_VERSION = "2024-10-21"

# ---------------------------------------------------------------------------
# MCP (Model Context Protocol)
# ---------------------------------------------------------------------------

MCP_TOOL_PREFIX = "mcp"
MCP_DEFAULT_TIMEOUT = 30.0
MCP_TRANSPORT_STDIO = "stdio"
MCP_TRANSPORT_SSE = "sse"
MCP_TRANSPORT_STREAMABLE_HTTP = "streamable-http"

# ---------------------------------------------------------------------------
# Agent handoff
# ---------------------------------------------------------------------------

DEFAULT_MAX_HANDOFF_DEPTH = 3
HANDOFF_CONTEXT_MAX_MESSAGES = 20
HANDOFF_MESSAGE_TRUNCATE_LENGTH = 500
HANDOFF_MESSAGE_TRUNCATE_SUFFIX_AT = 497
HANDOFF_TOOL_NAME = "handoff"

# ---------------------------------------------------------------------------
# Intelligent routing
# ---------------------------------------------------------------------------

ROUTING_TRIVIAL_MAX_LEN = 100
ROUTING_SIMPLE_MAX_LEN = 300
ROUTING_MODERATE_MAX_LEN = 600
ROUTING_COMPLEX_MAX_LEN = 1000
ROUTING_TRIVIAL_MAX_DEPS = 0
ROUTING_SIMPLE_MAX_DEPS = 1
ROUTING_MODERATE_MAX_DEPS = 3
