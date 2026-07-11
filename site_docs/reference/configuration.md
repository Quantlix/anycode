---
title: "AnyCode Configuration Reference"
description: "Every configuration surface in AnyCode — OrchestratorConfig and its nested feature configs, AgentConfig, TeamConfig, TaskSpec, RunnerOptions, resilience and streaming settings, and notable defaults."
keywords: AnyCode configuration, OrchestratorConfig, AgentConfig, TeamConfig, TaskSpec, RunnerOptions, MemoryConfig, CheckpointConfig, RetryPolicy, defaults
---

# Configuration

The engine accepts `AnyCode(config=...)` as an `OrchestratorConfig`, a plain `dict` (validated into one), or `None`. All configuration models are frozen Pydantic models — invalid keys or values raise a `ValidationError` at construction, not at run time.

```python
from anycode import AnyCode, OrchestratorConfig

engine = AnyCode(config={"max_concurrency": 3, "default_provider": "anthropic"})
# equivalent:
engine = AnyCode(config=OrchestratorConfig(max_concurrency=3, default_provider="anthropic"))
```

## `OrchestratorConfig`

### Core

| Key | Type | Default | Controls |
| --- | --- | --- | --- |
| `max_concurrency` | `int \| None` | `None` → `5` | Max agents running in parallel |
| `default_model` | `str \| None` | `None` | Fallback model when an agent omits one |
| `default_provider` | `str \| None` | `None` | One of `anthropic`, `openai`, `google`, `ollama`, `bedrock`, `azure` |
| `on_progress` | `Callable \| None` | `None` | Callback receiving each `OrchestratorEvent` |
| `max_handoff_depth` | `int` | `3` | Max chained agent handoffs |
| `handoff_policy` | `HandoffPolicy \| None` | `None` | Auto-handoff decision policy |
| `mcp_servers` | `list[MCPServerConfig] \| None` | `None` | MCP servers to connect — see [Connect MCP servers](../guides/mcp.md) |
| `verification` | `tuple[VerificationSensorConfig, ...]` | `()` | Team-level quality sensors |
| `approval_handler` | `ApprovalGate \| None` | `None` | Gate implementation; required for approval to activate |

### Feature configs

Each is `None`/disabled by default; set it to switch the feature on.

**`memory: MemoryConfig`** — `backend` (`"memory"` \| `"sqlite"` \| `"redis"`, default `"memory"`), `path`, `url`, `vector_backend` (`"none"` \| `"memory"` \| `"chromadb"`, default `"none"`), `vector_path`, `redact_sensitive_data=True`.

**`checkpoint: CheckpointConfig`** — `enabled=False`, `backend` (`"filesystem"` \| `"sqlite"`, default `"filesystem"`), `path=".anycode/checkpoints"`, `keep_last=5`, `redact_sensitive_data=True`.

**`approval: ApprovalConfig`** — `enabled=False`, `timeout_seconds=300.0`, `default_on_timeout` (`"approve"` \| `"reject"`, default `"reject"`), `require_approval_tools=None`, `require_approval_tasks=False`.

**`routing: RoutingConfig`** — `enabled=False`, `rules: list[RoutingRule]`, `default_model`, `default_provider`, `classify_with_llm=False`. Each `RoutingRule` has `condition`, `target_model`, `target_provider`, `priority=0`.

**`cost: CostConfig`** — `enabled=True`, `budget_usd=None`, `alert_threshold=0.8`, `on_budget_exceeded` (`"stop"` \| `"warn"` \| `"continue"`, default `"stop"`), `custom_pricing`.

**`reflection: ReflectionConfig`** — `enabled=False`, `mode` (`"self"` \| `"peer"` \| `"custom"`), `critic_model`, `critic_provider`, `quality_threshold=0.7`, `max_reflections=2`, `critic_prompt`, `custom_critic`.

**`rag: RAGConfig`** — `enabled=False`, `auto_index=True`, `top_k=5`, `min_relevance=0.3`, `max_context_tokens=2000`, `index_tool_results=True`, `namespace="default"`.

Tracing and guardrails are supplied via `AnyCode.configure(trace=TraceConfig(...), guardrails=GuardrailConfig(...))` rather than the constructor:

**`TraceConfig`** — `enabled=False`, `service_name="anycode"`, `exporter` (`"otlp"` \| `"console"` \| `"none"`, default `"console"`), `endpoint`, `sample_rate=1.0`, `redact_sensitive_data=True`.

**`GuardrailConfig`** — `max_tokens_per_agent`, `max_tokens_per_team`, `max_cost_usd`, `max_turns`, `max_tool_calls` (all `None`), plus `blocked_tools`, `require_approval_tools`, `output_validators`.

## `AgentConfig`

| Field | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | required | Agent identifier |
| `model` | `str` | required | LLM model id |
| `provider` | `str \| None` | `None` | Provider override (same literals as above) |
| `system_prompt` | `str \| None` | `None` | System prompt |
| `tools` | `list[str] \| None` | `None` | Allowed tool names — see [Built-in tools](built-in-tools.md) |
| `max_turns` | `int \| None` | `None` → `10` | Turn cap |
| `max_tokens` | `int \| None` | `None` → `4096` | Output token cap |
| `temperature` | `float \| None` | `None` | Sampling temperature |
| `mcp_servers` | `list[str] \| None` | `None` | Which MCP servers' tools this agent sees |
| `context_policy` | `ContextPolicy \| None` | `None` | Per-agent context management |
| `verification` | `tuple[VerificationSensorConfig, ...]` | `()` | Per-agent quality sensors |
| `tool_security` | `ToolSecurityPolicy \| None` | `None` | Runtime tool, path, shell, and environment restrictions |

!!! note "Reasoning options live on `RunnerOptions`, not `AgentConfig`"
    `reasoning_effort` and `thinking_budget_tokens` are runner/chat options — see [Use reasoning models](../guides/reasoning-models.md).

## `TeamConfig` and `TaskSpec`

**`TeamConfig`** — `name: str` (required), `agents: list[AgentConfig]` (required), `shared_memory: bool | None`, `max_concurrency: int | None` (per-team parallelism cap), `memory_store: MemoryStore | None` (inject your own store).

**`TaskSpec`** — `title: str` (required), `description: str` (required), `assignee: str | None` (agent name), `depends_on: list[str] | None` (dependency titles; tasks run in dependency-ordered concurrent waves).

## `RunnerOptions`

Options for `AgentRunner`, the low-level per-agent loop:

| Field | Purpose |
| --- | --- |
| `model`, `agent_name` | Identity |
| `max_turns`, `max_tokens`, `temperature` | Loop and sampling caps |
| `reasoning_effort`, `thinking_budget_tokens` | [Reasoning models](../guides/reasoning-models.md) |
| `streaming: RunnerStreamingConfig` | [Streaming](../guides/streaming.md) — `enabled=True`, `fallback_to_chat=True` by default |

Related runtime configs:

**`ProviderResilienceConfig`** — `enabled=True`, `retry: RetryPolicy`, `circuit_failure_threshold=5`, `circuit_reset_seconds=120.0`, `enable_prompt_cache=True`.

**`RetryPolicy`** — `max_attempts=6`, `base_delay_seconds=1.0`, `max_delay_seconds=60.0`, `jitter=True`, `respect_retry_after=True`, `call_timeout_seconds=300.0`.

**`DurabilityConfig`** — `enabled=False`, `run_root=".anycode/runs"`, `checkpoint_every_turns=5`, `keep_last_checkpoints=3`, `heartbeat_seconds=30.0`, `redact_sensitive_data=True`.

**`RunRetentionPolicy`** — `max_age_days=None`, `max_runs=None`, `statuses=("completed", "failed", "cancelled")`. Pass it to `sweep_once(..., retention_policy=...)` or `RunScheduler(..., retention_policy=...)`; without a policy, runs are never pruned.

**`FilesystemRunStore`** — accepts `redact_sensitive_data=True` and an optional `payload_protector: RunPayloadProtector`. The protector secures serialized payload bytes; key management and storage metadata protection remain the application's responsibility. Inject any backend satisfying `RunStore` through `AgentRunner(run_store=...)`.

Durable run artifacts use schema format v1; workflow checkpoints use v2 and retain v1 read compatibility. Catch `UnsupportedRunStoreVersionError` or `UnsupportedCheckpointVersionError` to stop startup and run a controlled migration when storage was written by a newer AnyCode release.

**`ContextPolicy`** (per agent) — thresholds that trigger context management as the window fills: `max_context_tokens=100_000`, then ratios `trim_ratio=0.65`, `mask_ratio=0.70`, `offload_ratio=0.75`, `compact_ratio=0.85`, `handoff_ratio=0.95`; plus `keep_recent_messages=6`, `max_tool_output_tokens=4000`, `summary_target_tokens=800`, `artifact_dir=".anycode/artifacts"`, `redact_sensitive_data=True`, `mode` (`"disabled"` \| `"manual"` \| `"auto"`).

The `redact_sensitive_data` flags scrub recognized credentials before telemetry export or persistence. Leave them enabled unless the destination is independently protected and exact replay is required. They are pattern- and key-based defenses, not a substitute for encryption, access control, data classification, or retention policies.

## Notable defaults

| Constant | Value | Meaning |
| --- | --- | --- |
| `DEFAULT_MAX_CONCURRENCY` | `5` | Agent pool size |
| `DEFAULT_TOOL_CONCURRENCY` | `4` | Parallel tool calls per turn |
| `DEFAULT_TURN_LIMIT` | `10` | Agent loop turns |
| `DEFAULT_MAX_TOKENS` | `4096` | Output tokens per call |
| `BASH_TIMEOUT_LIMIT_S` | `30` | `bash` tool timeout |
| `BASH_MAX_OUTPUT_BYTES` | `200_000` | `bash` output cap per stream |
| `GREP_MATCH_CEILING` | `100` | `grep` result cap |
| `LIST_FILES_CEILING` | `1000` | `list_files` result cap |
| `MCP_DEFAULT_TIMEOUT` | `30.0` | MCP connect/call timeout |
| `OLLAMA_DEFAULT_BASE_URL` | `http://localhost:11434` | Local Ollama endpoint |
| `AZURE_DEFAULT_API_VERSION` | `2024-10-21` | Azure OpenAI API version |

## See also

- [Public API](public-api.md) — everything importable from `anycode`
- [Production controls](../guides/production-controls.md) — budgets, durability, verification in practice
