# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0] - 2026-07-24

### Added

- **Expanded sandbox provider catalog** - E2B, Modal, Runloop, Vercel Sandbox, and LangSmith backends now implement the `SandboxProvider` protocol alongside Daytona, each behind its own install extra (`sandbox-e2b`, `sandbox-modal`, `sandbox-runloop`, `sandbox-vercel`, `sandbox-langsmith`) with lazy SDK imports, honest capability reports, evidence digests, and fail-closed handling of unsupported network modes, snapshot restores, and secret schemes. A new `create_sandbox_provider(name)` factory builds any backend by name.
- **Robust Ollama integration** - the Ollama adapter now supports thinking (`reasoning_effort`/`thinking_budget_tokens` map to the native `think` parameter, with `ThinkingBlock` content and `thinking` stream events), structured outputs via `chat(..., response_format=...)` translated to Ollama's `format`, sampling options (`max_tokens` → `num_predict`, plus a `default_options` passthrough for `seed`, `top_p`, `num_ctx`, and friends), `keep_alive`, and ollama.com cloud authentication through `OLLAMA_API_KEY` or `api_key=`.
- **Sandbox and Ollama examples** - `examples/41_sandbox_catalog.py` (offline provider catalog, capability reports, fail-closed guards), `examples/42_vercel_sandbox.py` and `examples/43_modal_sandbox.py` (full lifecycle verified against live Vercel and Modal sandboxes), and `examples/44_ollama_robustness.py` (thinking, structured outputs, streaming, tool calls, and error handling verified against a live Ollama server).

### Changed

- **Provider-prefixed sandbox secrets** - `SandboxSpec.secret_references` now accepts any `<provider>:<name>` reference instead of only `daytona:`; each backend validates its own prefix at create time and returns `sandbox_secret_reference_invalid` for foreign prefixes. Existing `daytona:` references keep working unchanged.

### Fixed

- **Ollama image input and error reporting** - image blocks are now sent as the native base64 `images` array (previously an unsupported `image_url` shape the server ignored), mid-stream NDJSON error objects surface as terminal `error` stream events instead of a silently truncated answer, `done_reason: "length"` maps to `stop_reason="max_tokens"`, and a `404` names the missing model with the exact `ollama pull` command.
- **Vercel sandbox on Windows** - the Vercel SDK imports Unix-only pty modules (`termios`/`tty`) at import time for an interactive-shell helper the adapter never calls; the adapter now stubs them on Windows so the sandbox API loads instead of failing with `ModuleNotFoundError`.
- **Modal filesystem API** - sandbox file transfer now prefers `Sandbox.filesystem.write_bytes`/`read_bytes` over the deprecated `Sandbox.open()`, falling back to `open()` on older Modal SDKs.

## [0.8.2] - 2026-07-19

### Changed

- **Runnable, searchable documentation** - added complete copyable programs across core guides, expanded CLI and public API coverage, normalized page descriptions for search snippets, removed duplicate social metadata, and added regression checks for page structure and version-switcher deployment.

### Fixed

- **Release metadata recovery** - synchronized package and documentation release metadata for the required `v0.8.2` tag after the malformed `0.8.1` GitHub release failed before publishing a package or versioned documentation.

## [0.8.0] - 2026-07-16

### Added

- **Portable agent infrastructure preview** - added pluggable in-memory, SQLite, and Dapr durability backends with leases, fencing, signals, migration, conformance, and failure-soak coverage; execution identity and fail-closed external policy enforcement; versioned OpenTelemetry GenAI mapping and capture profiles; companion and Daytona sandbox adapters; policy-constrained multi-provider routing; a browser/Node TypeScript service client; and container/Kubernetes hosting profiles with graceful drain and endpoint-specific A2A Agent Cards.
- **Versioned semantic contract preview** - added strict JSON-only models and checked-in schemas for runs, tasks, messages, artifacts, events, checkpoints, policy decisions, verification results, and capability descriptors. Deterministic state, cancellation, retry, dependency, resume, projection, leased claim/fencing, and artifact-integrity semantics include golden histories, exhaustive state checks, race coverage, and a credential-free end-to-end example.
- **Maintainer and contributor governance** - added authoritative maintainer, contribution, security-reporting, and release policies covering roles, branches, review evidence, compatibility, deprecation, versioning, backports, Trusted Publishing, release verification, and recovery. Versioned site guides expose the development, governance, and release workflows to contributors.
- **Source-linked documentation validation** - added a generated inventory for every package-root public export and `scripts/check_docs.py` checks for API coverage, registered built-in tools, numbered examples, page metadata, and curated `llms.txt` links. Documentation CI and package publication gates now run the strict site build and consistency check.
- **Executable runtime contract and baseline** - documented the current capability matrix, lifecycle transition table, verification attachment points, persisted local formats, supported resume scenarios, side-effect boundary, ADR template, and contract-test conventions. A deterministic example now records task admission, execution, checkpoint size, event volume, and context growth in local or CI evidence, and a real child-process exit test proves cleanup-independent durable resume.

### Changed

- **Responsive documentation and discovery** - rebuilt the documentation home and content layouts for mobile navigation, narrow code blocks, scrollable tables, accessible focus states, and balanced desktop grids; removed promotional hero badges; corrected duplicate canonical and heading metadata; added dedicated durability-backend, execution-identity, policy-routing, sandbox-provider, service-hosting, and GenAI-telemetry guides; and synchronized the README, `llms.txt`, release runbook, feature guides, and TypeScript client coverage.
- **Release-bound documentation publishing** - pushes to `main` now validate documentation without overwriting released pages. Final release tags publish the `X.Y` docs and move `latest`; pre-release tags publish a candidate version without moving `latest`. Package publishing resolves locked dependencies and runs repository-wide quality, test, documentation, build, and metadata gates before Trusted Publishing.
- **Team verification lifecycle** - `run_team()` now evaluates `after_team` exactly once against coordinator and task output, preserves lifecycle and verification evidence on `TeamRunResult`, and returns a recoverable failure for team-level retry decisions. Passing tool-boundary gates return from `verifying` to `executing`, allowing `before_tool` and `after_tool` sensors to coexist in one legal lifecycle.

## [0.7.0] - 2026-07-11

### Added

- **MCP, plugin, and tool trust hardening** - agent tool allowlists are now enforced again at execution, including an explicit empty list, so a provider cannot invoke an unadvertised registered tool. MCP tools require exact per-agent server opt-in, remain bound to discovery ownership across prebuilt agents and reconnects, always use side-effect idempotency, fail closed when configured auth is missing, and clean up partial initialization without masking cancellation. Plugin entry points are filtered before import and plugin contributions are preflighted before shared registry mutation. The security threat model and production-readiness checklist define the remaining host, network, identity, storage, and operational responsibilities.
- **Cross-platform compatibility CI** - pull requests and `main` now run the complete non-integration suite on Linux, Windows, and macOS across Python 3.12 and 3.13. Separate jobs enforce locked quality checks, core-only and per-extra dependency isolation, Redis/ChromaDB integration coverage, and built wheel smoke tests for both core and CLI installations. Repository tests keep the optional-extra matrix synchronized with package metadata.
- **Enforced compatibility contracts** - the complete v0.6 top-level Python API now has an additive CI baseline, and duplicate public declarations fail tests. Declarative YAML/TOML files use format v1, preserve unversioned v1 compatibility, and reject future versions with `UnsupportedConfigVersionError`. The compatibility reference defines public import boundaries, semantic-version rules, checkpoint and durable-run reader ranges, plus upgrade and rollback procedures.
- **Operational observability** - agent runs now share durable run/trace correlation across turn, LLM, tool, and terminal spans, with task-local async parenting and deterministic per-trace sampling. Completed spans automatically feed bounded latency, first-token, token, estimated-cost, retry, outcome, and error metrics plus redacted structured events. The new `jsonl` exporter emits one correlated completion record per sampled span for container log collectors; OTLP spans preserve runtime timing, carry explicit AnyCode correlation attributes, and expose flush/shutdown lifecycle controls. Configurable span/event/series/histogram retention prevents long-lived telemetry growth and exposes drop counters, while exporter failures remain isolated from run behavior.
- **End-to-end cancellation ownership** — caller cancellation now propagates through `AgentRunner`, high-level run and stream APIs, orchestrator waves, provider waits, parallel tools, and shell process trees without being converted into an ordinary result. Agents settle in an explicit `cancelled` state, durable runs persist a `user_cancelled` stop and checkpoint, semaphore accounting remains balanced, and `AnyCode.close()` cancels and awaits all tracked standalone, coordinator, team, reflection, and handoff operations before resource teardown.
- **Fail-closed side-effect idempotency** — tools can opt into atomic claim-before-execute semantics with `side_effecting=True`. Explicit business keys take precedence over deterministic run/turn/call fallbacks; completed calls replay, mismatched input conflicts, and in-progress or unrecorded outcomes terminate the run with `side_effect_unknown` instead of being retried. Post-invocation errors default to non-retryable unless the tool explicitly proves otherwise, and uncertain claims are retained during pruning for operator reconciliation. Public in-memory and SQLite stores support process-local or restart-safe coordination, hashed storage keys, redacted result persistence, and pruning. Mutating built-ins and every discovered MCP tool are protected by default.
- **Provider capacity controls** — `ResilientAdapter` now applies a provider-scoped concurrency bulkhead (default `8`) and optional evenly paced `requests_per_minute` limit to chat and streaming attempts, including retries. Capacity is shared across adapters in the same event loop and scope; conflicting limits for one scope fail clearly. Queue waits load-shed with `ProviderCapacityError` after a configurable timeout, while cancellation and early stream closure release slots immediately. `ProviderResilienceConfig` is configurable globally through `OrchestratorConfig`/YAML or per `AgentConfig`, with per-agent precedence.
- **Protected and bounded durable storage** — `AgentRunner` and `RunScheduler` now accept the public `RunStore` protocol so production backends can replace the local filesystem implementation. `FilesystemRunStore` accepts a `RunPayloadProtector` for versioned, fail-closed protected payload envelopes while retaining read compatibility with legacy plaintext stores. Run records, transcript events, and turn checkpoints now carry an explicit schema version; legacy unversioned artifacts remain readable and unsupported future versions fail clearly. Workflow checkpoints now default correctly to format v2 while retaining v1 compatibility. `RunRetentionPolicy` prunes only terminal runs by age and count, and can be applied through `sweep_once`, `RunScheduler`, or `anycode runs sweep`. Run IDs are constrained to one path segment to prevent storage-root traversal.
- **Default-on credential redaction** — centralized `redact_text`, `redact_sensitive`, and `safe_exception_message` helpers now protect built-in telemetry exports, exception surfaces, workflow and turn checkpoints, run records and transcripts, context artifacts, session-chain files, persistent memory backends, eval reports, and harness artifacts. Structured sensitive keys and common provider/cloud token formats are replaced with `<redacted-secret>` while token-usage metrics and other non-secret fields retain their shape. Persistence and exporter configs expose explicit `redact_sensitive_data=False` opt-outs for independently protected stores that require exact replay.

## [0.6.0] - 2026-07-10

### Added

- **Provider resilience & prompt caching** — new `anycode.providers.resilience` module: `ResilientAdapter` wraps any `LLMAdapter` with classified retry/backoff (429/5xx/timeouts/connection errors, honoring `Retry-After`), a wall-clock deadline per call, and a per-provider circuit breaker that fails fast with `ProviderUnavailableError` while open. `create_adapter` wraps every built-in and plugin provider by default (`ProviderResilienceConfig(enabled=False)` opts out). New `provider_unavailable` stop reason surfaces exhausted retries as a structured, recoverable `RunResult` instead of a raw error. The Anthropic adapter now requests prompt caching (`cache_control` breakpoints on the stable system+tools prefix) whenever the resolved model profile supports it. New `tokens` extra declares `tiktoken` so token accounting can upgrade past the chars/4 heuristic. Tests: `tests/test_resilience.py`.
- **Durable run store & mid-run checkpoints** — new `anycode.runstore` package: one directory per run with an atomic `meta.json` (`RunRecord` — status, heartbeat, wake condition), an append-only `transcript.jsonl` event log (`TranscriptEvent`, torn tails tolerated), and pruned `TurnCheckpoint`s carrying full conversation, budget, cost, loop-detector window, lifecycle events, verification results, gate decisions, and context manifests. `AgentRunner` accepts `durability=DurabilityConfig(...)` (opt-in; default behavior unchanged) plus `resume_from=` to continue a killed run from its last turn boundary with accounting intact, and `BudgetTracker`/`LoopDetector` gained snapshot/restore. `CheckpointData` bumped to format v2: serialized agent results now retain lifecycle/verification/gate/manifest state (v1 files still load). Team workflows checkpoint after every completed task, so a mid-wave crash resumes at the first incomplete task instead of re-running the wave. Tests: `tests/test_runstore.py`.
- **Automatic context reset & session chaining** — handoff artifacts upgraded to a five-layer structure (typed state, narrative, decisions, next steps, warnings) and, with `ContextPolicy(auto_reset_on_handoff=True)`, the runner now rebuilds its conversation from the artifact mid-run at `handoff` pressure — same run identity, budget, and audit trail, fresh window — re-injecting task-state invariants as a maximally recent message. New `GoalContract`/`GoalCriterion` (criteria flip only through an external verifier, never the agent's own claim) and `SessionChain` (`anycode.core.session_chain`), which drives fresh-context sessions over a persisted contract plus append-only `progress.md`. Tests: `tests/test_session_chain.py`.
- **Tiered persistent memory** — the orchestrator now honors `MemoryConfig.vector_backend` through the new `create_vector_store` factory instead of hardcoding the in-memory TF-IDF store, so RAG memory survives restarts with `vector_backend="chromadb"` (a loud warning fires when long-term memory is volatile). New `anycode.memory.knowledge` module: `KnowledgeStore` persists curated "what was learned" entries as human-editable Markdown+frontmatter files with provenance (source, author, timestamp, content hash) and append-plus-supersede curation; `build_knowledge_tools` exposes opt-in `knowledge_save`/`memory_search` agent tools; `apply_retention` gives rolling logs FIFO retention. Tests: `tests/test_knowledge.py`.
- **Context lifecycle hardening** — new `mask` pressure stage (between `trim` and `offload`) replaces aged tool results with short restorable pointers while protecting the recency window. Compaction is now archive-first: the untouched history is written to disk before any summarization and the archive path plus an artifact index are injected into the summary, so compaction is never lossy (`ContextManifest.archive_path`). `ContextManager` accepts an optional `summarizer` callable (e.g. an LLM) with the deterministic extractive path as both default and failure fallback, re-injects preserved task-state invariants after every compaction boundary, and calibrates pressure classification against provider-actual token counts via `note_actual` (EMA, clamped). Tests: `tests/test_context_hardening.py`.
- **Scheduling, heartbeats & watchdogs** — runs can now pause with a persisted `WakeCondition` (`at_time`, `on_approval`, `on_provider_recovery`, `manual`) and be woken by an idempotent, concurrency-safe sweep (`sweep_once`, per-run lock with stale takeover) from cron or the in-process `RunScheduler` tick loop. Watchdog semantics separate liveness from progress: stale heartbeat → `interrupted` (crash), fresh heartbeat without progress → a `stall_warning` audit event, never an automatic kill. A durable run that exhausts provider retries now pauses with a timed `on_provider_recovery` wake instead of failing. New `anycode.schedule` package also ships `ScheduledTask` modes (`notification`/`script`/`agent`/`hybrid`) so recurring work spends tokens on judgment, not mechanics. Tests: `tests/test_schedule.py`.
- **Runs operator CLI** — new `anycode runs` command group over the run store: `list` (status/turns/cost), `show` (record, wake condition, accounting, recent events), `tail` (events after a sequence number), `audit` (deterministic digest of a time window: event counts, tools used, stops/pauses/stalls), and `sweep` (one watchdog pass). All views derive from the same append-only transcript the runner writes. Tests: `tests/test_runs_cli.py`.
- New examples: `examples/28_durable_runs.py` (kill-and-resume), `examples/29_session_chain.py` (goal contract across fresh contexts), `examples/30_scheduled_wakeups.py` (pause/wake sweeps + scheduled task modes) — all runnable with `FakeAdapter`, no API keys.
- **Plugin / extension ecosystem** — new `anycode.plugins` package introducing `PluginManifest`, the `Plugin` Protocol, a `PluginBase` no-op default, and `PluginRegistry`. Plugins bundle custom tools, async provider factories, verification sensors, and turn hooks into a single object. `AnyCode.register_plugin(...)` installs a plugin into the engine; `AnyCode.load_installed_plugins()` discovers and installs every plugin published under the `anycode.plugins` entry-point group. `create_adapter` now dispatches unknown provider names through the plugin-registered provider-factory registry. `anycode inspect plugins` and the augmented `anycode inspect providers` surface discovered plugins and plugin-contributed providers. New example `examples/27_plugin_ecosystem.py` and tests `tests/test_plugins.py`.
- **Handoff chain recursion** — `HandoffExecutor.execute` now follows multi-hop chains: when the target agent itself emits a `handoff_request`, the executor recurses to the next agent up to `max_handoff_depth`, appending each hop (and any depth-limit short-circuit) to the optional `chain` argument. `AnyCode._run_wave_task` passes the chain list so `TeamRunResult.handoffs` records the full multi-hop path. YAML/TOML configs accept a top-level `max_handoff_depth` integer.
- **Context engineering for huge-context models** — new `anycode.context` package with a built-in `ModelContextProfile` registry (Anthropic 200k/1M, OpenAI 128k/1M, Google 1M/2M, plus unbounded fallback), profile resolution chain (`override → custom → built-in → provider default → unbounded`), and pluggable `Tokenizer` Protocol with heuristic + optional `tiktoken` backends. `ContextPolicy` gains `mode` (`disabled`/`manual`/`auto`), `reserved_response_tokens`, typed `sections` budgets, and `custom_profiles`/`model_profile` fields so any future window size — including 5M+ tokens — is supported without code changes. `ContextManager` now classifies content into `ContextSectionKind`s, applies per-section `overflow` strategies (`trim`/`summarize`/`offload`/`drop`/`error`), emits a `ContextUsageReport` on every manifest, and exposes `ContextManager.reconcile(...)` so provider-actual token counts replace heuristic estimates after the call. `TokenUsage` adds `cache_creation_input_tokens` and `cache_read_input_tokens`; `CostTracker`/`calculate_cost` bill cache reads at `cached_input_cost_per_1k` when available. The Anthropic adapter extracts cache token classes natively; the YAML/TOML config loader recognises a top-level `context_engineering` block plus per-agent `context_policy` overrides. New helpers `format_usage_report` and `render_usage_report_table` ship Markdown-ready summaries. New example `examples/26_context_engineering.py`, config `examples/config/context_engineering.yaml`, doc `docs/context-engineering.md`, and tests `tests/test_context_engineering.py`, `tests/test_model_profiles.py`, `tests/test_token_accounting.py`, `tests/test_huge_context.py`.
- **Runtime telemetry & cancellation** — `AgentRunner.stream` now catches `asyncio.CancelledError`, emits a terminal `cancelled` lifecycle phase with a `user_cancelled` `StopReason`, and records final `phase`/`stop_reason`/`recoverable` attributes on a dedicated `anycode.agent.{name}.terminal` span before re-raising.
- **Adaptive context lifecycle** — `ContextPolicy` gains `provider_overrides`, `preserved_task_state`, and `preserved_verification_failures`. `ContextManager` accepts a `provider=` kwarg, resolves the matching override via `ContextPolicy.for_provider()`, and emits a `ContextManifest` that includes the resolved provider plus preserved state/failure sections during compaction.
- **Declarative quality gates** — new `anycode.verification.registry` exposes `register_sensor_factory`, `build_sensor`, and `build_sensors`. Built-in factories cover `ruff`, `pyright`, `pytest`, and a pure-Python `regex` sensor. `AgentConfig.verification`, `RunnerOptions.verification`, and `OrchestratorConfig.verification` plumb `VerificationSensorConfig` tuples. The runner instantiates a `QualityGate` and evaluates it at `before_tool`, `after_tool`, and `after_task` phases; the orchestrator builds a separate team-level gate and evaluates it at `after_team`. `block`/`escalate` outcomes translate into a `verification_failed` `StopReason` on `AgentRunResult` and `TeamRunResult`, while `retry` outcomes feed sensor feedback back into the agent loop. The YAML config loader reads top-level and per-agent `verification:` blocks.
- **Team lifecycle aggregation** — `TeamRunResult` now exposes aggregated `lifecycle_events`, `verification_results`, `gate_decisions`, and a top-level `stop_reason` so callers can inspect every task's lifecycle trail and any team-level gate outcome.
- **Deterministic evaluation suite** — new `anycode.providers.fake.FakeAdapter` (and `FakeResponse`) replays a scripted reply sequence with no LLM credentials. `EvalScenario` gains `deterministic`, `fake_responses`, and `fake_tool_failures` fields; `run_scenario` now branches into a deterministic harness when the flag is set. `EvalScenarioResult` and `EvalReport` aggregate `cost_usd`, `retries`, and `verification_failures` so CI can track the new metrics.
- New examples: `examples/22_deterministic_eval.py`, `examples/23_context_pressure.py`, `examples/24_verification_gates.py`, `examples/25_runtime_cancellation.py`.
- New deterministic eval fixture `tests/fixtures/eval/runtime_reliability_deterministic.yaml`.
- New tests in `tests/test_harness_runtime.py` covering provider overrides, the sensor registry, the deterministic suite, runner cancellation telemetry, multi-phase quality gates (`before_tool`/`after_tool`), and orchestrator team-level (`after_team`) gating.

### Changed

- Declarative config loading now rejects unknown root, agent, task, context-engineering, and nested model fields with `UnknownConfigFieldError` instead of silently ignoring them. Programmatic model construction is unchanged.
- **Handoff sentinel encoding** — the built-in `handoff` tool now encodes its payload as `__HANDOFF__:<json>` instead of a colon-delimited string. Free-form `summary`/`reason` text containing `:` (or any other character) now round-trips losslessly through `AgentRunner._detect_handoff`. New helpers `encode_handoff_payload`/`decode_handoff_payload` are exported from `anycode.handoff.tool`.
- **`AgentState`** uses `Field(default_factory=...)` for `messages` and `token_usage` to guarantee per-instance defaults.
- **`AgentRunner.stream` handoff path** now appends every executed `ToolCallRecord` from the same batch and emits a `tool_result` event for each before yielding the `handoff`/`done` events. Previously, sibling tool calls executed alongside a handoff were dropped from `RunResult.tool_calls` and never streamed to observers.

### Fixed

- `validate_config()` no longer reports a missing `system_prompt` as a validation error — it was only ever a soft recommendation, so callers like `anycode inspect config` no longer fail hard on configs that omit it.
- The PyYAML `ImportError` raised by `anycode.config.loader` now points at the correct extras: `pip install "anycode-py[cli]"` (the package on PyPI is `anycode-py`, not `anycode`).

## [0.5.0] - 2026-05-06

### Added

- **Agent Handoff (orchestrator integration)** — `Team` workflows now route handoff requests through `HandoffExecutor`, validate handoff targets against team membership, support policy-driven handoffs via `OrchestratorConfig.handoff_policy`, and emit `Handoff` records on `TeamRunResult.handoffs`.
- **Intelligent Routing (orchestrator integration)** — `Router.route()` decisions are applied per task before execution; the resolved model/provider override is layered onto the agent config without mutating the original. `TeamRunResult.route_decisions` exposes the decision trail.
- **CLI Toolkit** — new `anycode` CLI (built on `typer` + `rich`):
  - `anycode init <dir>` scaffolds a project (`team.yaml`, `main.py`, `.env.example`, `tools/`, `.gitignore`).
  - `anycode run <config.yaml>` loads a team config and runs it end-to-end.
  - `anycode inspect tools|providers|team <path>|config <path>` introspects the runtime.
  - `anycode version` prints package + Python info.
  - Available via `pip install anycode-py[cli]`.
- **Declarative YAML/TOML config** (`src/anycode/config/`) — `load_config(path)` + `validate_config(path)` parse `.yaml`/`.yml`/`.toml` files into typed `LoadedConfig` with `${ENV_VAR}` substitution. New `AnyCode.from_config(path)` and `engine.run_team_from_config(goal=...)` classmethod/method.
- **Examples cookbook** — five new end-to-end examples (`13_cost_tracking.py`, `14_self_reflection.py`, `15_rag_memory.py`, `16_dag_visualization.py`, `17_yaml_config.py`).
- **Self-Reflection / Critic Loop** (`src/anycode/reflection/`) — `LLMCritic`, `parse_critic_json`, `ReflectionLoop` with `self`/`peer`/`custom` modes. Configured via `OrchestratorConfig.reflection = ReflectionConfig(...)`. Tracks `reflections_count` and `quality_score` on `AgentRunResult`.
- **Cost-Aware Execution Engine** (`src/anycode/cost/`) — `CostTracker`, `build_cost_report`, `DEFAULT_PRICING`, `find_pricing` (with wildcard fallback), `calculate_cost`. Configured via `OrchestratorConfig.cost = CostConfig(budget_usd=..., on_budget_exceeded="stop"|"warn"|"continue")`. Emits cost-alert events at the configured threshold and stops execution when the budget is exhausted (when `on_budget_exceeded="stop"`). `TeamRunResult.cost_report` exposes per-agent and per-model breakdown.
- **DAG Visualization** (`src/anycode/viz/`) — `render_dag(queue, format="mermaid"|"dot"|"json"|"ascii", show_status=True)` and `render_timeline(team_result, width=40)`. Mermaid output includes `classDef` styling per task status.
- **RAG Memory** (`src/anycode/memory/rag.py`, `src/anycode/memory/indexer.py`) — `RAGRetriever` (dedup, namespace filtering, relevance/token caps) and `RAGIndexer` (paragraph-aware chunking, optional tool-result indexing). Configured via `OrchestratorConfig.rag = RAGConfig(...)`. RAG context is auto-injected into every task prompt and outputs are auto-indexed to the configured `VectorStore` (defaults to `InMemoryVectorStore`).

### Changed

- `AgentRunResult` gained `handoff_request`, `reflections_count`, and `quality_score` fields (all optional / defaulted).
- `RunResult` gained `handoff_request` field (optional).
- `TeamRunResult` gained `handoffs`, `route_decisions`, and `cost_report` fields (all optional).
- `OrchestratorConfig` gained `cost`, `reflection`, and `rag` fields (all optional).
- Public exports added to `anycode.__init__`: `CostTracker`, `build_cost_report`, `DEFAULT_PRICING`, `calculate_cost`, `find_pricing`, `render_dag`, `render_timeline`, `LLMCritic`, `ReflectionLoop`, `parse_critic_json`, `RAGRetriever`, `RAGIndexer`, plus the new types `ModelPricing`, `CostConfig`, `CostBreakdown`, `CostReport`, `CriticResult`, `Critic`, `ReflectionConfig`, `RAGConfig`, `RAGContext`, `RAGEntry`.

### Tests

- 43 new tests across `tests/test_cost.py`, `tests/test_viz.py`, `tests/test_config.py`, `tests/test_reflection.py`, `tests/test_rag.py`, `tests/test_cli.py`. Total suite: 343 passing.

## [0.4.0] - 2026-06-10

### Added

- **Additional LLM Providers** — 4 new provider adapters implementing the `LLMAdapter` Protocol.
  - `GeminiAdapter` — Google Gemini via `google-genai` SDK with function calling and streaming support.
  - `OllamaAdapter` — Local Ollama models via HTTP (`httpx`), zero external SDK dependencies, OpenAI-compatible tool format.
  - `BedrockAdapter` — AWS Bedrock for Claude models via `boto3`, Anthropic message format, streaming via response streams.
  - `AzureOpenAIAdapter` — Azure OpenAI via the official `openai` SDK with Azure-specific auth and deployment configuration.
  - `_openai_compat` shared helper module — extracted common OpenAI mapping logic (messages, tools, stop reasons) for reuse across OpenAI, Azure, and Ollama adapters.
  - Extended `create_adapter()` factory to resolve all 6 providers with lazy imports.
- **MCP Integration module** (`src/anycode/mcp/`) — Model Context Protocol support for external tool servers.
  - `MCPClient` — manages connection lifecycle (stdio, SSE, streamable-http transports) via the official `mcp` SDK, with tool discovery and tool execution.
  - `schema_to_pydantic_model()` — dynamic Pydantic model generation from JSON Schema for MCP tool inputs.
  - `mcp_tool_to_definition()` — converts MCP tools into AnyCode `ToolDefinition` with prefixed naming (`mcp_{server}_{tool}`).
  - `discover_and_register()` — batch discovery and registration of MCP tools into the `ToolRegistry`.
  - `validate_server_config()` — transport-aware configuration validation.
  - `ToolRegistry.register_from_mcp()` and `ToolRegistry.deregister_prefix()` for MCP tool lifecycle management.
- **Agent Handoff module** (`src/anycode/handoff/`) — context-preserving agent-to-agent task delegation.
  - `HANDOFF_TOOL_DEF` — built-in sentinel tool that agents call to request a handoff (returns `__HANDOFF__:to:summary:reason`).
  - `HandoffExecutor` — orchestrates context transfer with conversation trimming, system/user prompt generation, and configurable depth limiting.
  - `trim_context()`, `build_handoff_system_prompt()`, `build_handoff_user_message()` — protocol helpers for handoff payloads.
  - Runner integration: `AgentRunner` detects handoff sentinels in tool results and yields `StreamEvent(type="handoff")`.
- **Intelligent Routing module** (`src/anycode/routing/`) — zero-cost heuristic task routing.
  - `classify_task()` — microsecond complexity classification (5 levels: trivial, simple, moderate, complex, expert) based on description length and dependency count.
  - `match_rule()` / `evaluate_rules()` — declarative rule engine supporting complexity conditions, keyword-in checks, and regex patterns with priority ordering.
  - `DefaultRouter` — `Router` Protocol implementation combining classifier + rules engine with default model fallback.
  - Orchestrator integration: routing decisions applied before task wave execution.
- New Pydantic types (all `frozen=True`): `MCPServerConfig`, `MCPToolInfo`, `HandoffRequest`, `Handoff`, `HandoffPolicy` Protocol, `ComplexityLevel`, `RoutingRule`, `RoutingConfig`, `RouteDecision`, `Router` Protocol.
- Extended `AgentConfig.provider` literal to include `"google" | "ollama" | "bedrock" | "azure"`.
- Extended `OrchestratorConfig` with `mcp_servers`, `handoff_policy`, `max_handoff_depth`, `routing` fields.
- Extended `TeamRunResult` with `handoffs` field.
- Optional dependency groups in `pyproject.toml`: `google` (`google-generativeai>=0.8`), `bedrock` (`boto3>=1.34`), `azure` (`openai>=1.50`), `mcp` (`mcp>=1.0`).
- **Examples**: `examples/09_multi_provider.py`, `examples/10_mcp_tools.py`, `examples/11_agent_handoff.py`, `examples/12_intelligent_routing.py`.
- **Test suites**: `tests/test_providers.py` (35 tests), `tests/test_mcp.py` (24 tests), `tests/test_handoff.py` (19 tests), `tests/test_routing.py` (18 tests).

### Changed

- **Orchestrator** — `AnyCode` now manages MCP client lifecycles (connect/disconnect) as an async context manager, registers the handoff tool for agents that opt in, injects per-agent MCP tools into the tool registry, and applies routing decisions before task wave execution.
- **AgentRunner** — detects handoff sentinel results in the tool loop; on detection, yields a `StreamEvent(type="handoff")` and terminates the turn.
- **ToolRegistry** — added `register_from_mcp()` for batch MCP tool registration and `deregister_prefix()` for cleanup on server disconnect.
- **providers/openai.py** — refactored to import shared mapping logic from `_openai_compat.py` (no behavior change).

## [0.3.0] - 2026-04-05

### Added

- **Pluggable Memory module** (`src/anycode/memory/`) — layered memory system with persistent KV stores and semantic vector search.
  - `SQLiteStore` — async SQLite-backed `MemoryStore` with WAL mode, metadata tracking, and `created_at`/`updated_at` timestamps.
  - `RedisStore` — Redis-backed `MemoryStore` for distributed deployments (optional `[redis]` extra).
  - `InMemoryVectorStore` — TF-IDF + cosine similarity vector search with zero external dependencies.
  - `ChromaDBVectorStore` — embedding-backed vector search via ChromaDB (optional `[vector]` extra).
  - `CompositeMemory` — unified interface querying both KV and vector stores with auto-indexing support.
  - `create_memory_store()` factory for config-driven backend creation from `MemoryConfig`.
- **Workflow Checkpointing module** (`src/anycode/checkpoint/`) — crash recovery for long-running DAG-based agent workflows.
  - `CheckpointManager` — automatic checkpoint creation after each execution wave, spec-change detection via SHA-256 hash, and configurable auto-pruning.
  - `FilesystemCheckpointStore` — human-readable JSON checkpoint files with atomic writes (tmp → rename).
  - `SQLiteCheckpointStore` — WAL-mode SQLite backend for high-concurrency checkpoint storage.
  - `serialize_checkpoint()` / `deserialize_checkpoint()` — deterministic round-trip serialization supporting all LLM message content types (`TextBlock`, `ToolUseBlock`, `ToolResultBlock`, `ImageBlock`).
- **Human-in-the-Loop module** (`src/anycode/hitl/`) — approval gates for enterprise-grade agent workflows.
  - `ApprovalManager` — config-driven approval enforcement with tool/task filtering and audit history tracking.
  - `CallbackApprovalGate` — programmatic approval via user-provided async callable.
  - `StdinApprovalGate` — interactive console approval with box-formatted prompts for CLI workflows.
  - `WebhookApprovalGate` — HTTP webhook + polling approval for async and remote approval flows.
  - `format_approval_request()` — box-formatted console output for approval prompts.
- New Pydantic types (all `frozen=True`): `VectorSearchResult`, `VectorStore` Protocol, `MemoryConfig`, `CheckpointConfig`, `CheckpointData`, `CheckpointStore` Protocol, `ApprovalConfig`, `ApprovalRequest`, `ApprovalResponse`, `ApprovalGate` Protocol.
- Optional dependency groups in `pyproject.toml`: `persistence` (`aiosqlite>=0.20`), `redis` (`redis[hiredis]>=5.0`), `vector` (`chromadb>=0.5`).
- **Examples**: `examples/06_pluggable_memory.py` (SQLite, Redis, vector search, composite memory, SharedMemory DI), `examples/07_checkpointing.py` (filesystem/SQLite stores, serialization, spec-change detection, crash/resume), `examples/08_hitl_approval.py` (callback/stdin/webhook gates, config enforcement, timeouts, audit trail).
- **Test suites** for the modules — unit tests (`test_memory.py`, `test_checkpoint.py`, `test_hitl.py`) and integration tests (`test_checkpoint_stores.py`, `test_composite_memory.py`, `test_full_pipeline.py`).

### Changed

- **Orchestrator** — `AnyCode` now saves checkpoints automatically after each execution wave via `CheckpointManager`, supports `resume_from` parameter (accepts `"latest"` or a specific checkpoint ID) for crash recovery, and enforces task-level approval gates via `ApprovalManager` before execution.
- **SharedMemory** — accepts any `MemoryStore` backend via constructor injection; defaults to `InMemoryStore` for full backward compatibility.
- **TeamConfig** — new optional `memory_store` parameter for pluggable team memory backends.
- **Types** — expanded `types.py` with 11 new Pydantic models for memory, checkpoint, and approval subsystems. All models remain frozen (immutable).

## [0.2.0] - 2025-04-05

### Added

- **Telemetry module** — OpenTelemetry-integrated tracing with `Tracer`, `Span`, and `ConsoleExporter` for full lifecycle visibility across agent runs. Includes `MetricsCollector` with `Timer` for latency tracking and `EventEmitter` for structured telemetry events.
- **Guardrails module** — Runtime safety layer with `BudgetTracker` for token/cost budget enforcement, `HookRunner` with `LoggingHook` for turn-level lifecycle hooks, and composable content validators (`MaxLengthValidator`, `ContainsValidator`, `BlocklistValidator`).
- **Structured output module** — Schema-constrained LLM responses via Pydantic models. Includes `schema_to_tool_def` and `schema_to_openai_response_format` for cross-provider schema conversion, `parse_structured_output` for validated extraction, and `build_retry_prompt` for automatic recovery on malformed responses.
- **Production features example** (`examples/05_production_features.py`) demonstrating telemetry, guardrails, and structured output working together in a real workflow.
- **Test suite** — Initial tests for guardrails, structured output, and telemetry modules.
- **Dev scripts** — `scripts/setup.sh`, `scripts/lint.sh`, `scripts/test.sh` for reproducible local development.
- `TraceConfig`, `SpanAttributes`, `GuardrailConfig`, `BudgetStatus`, `ValidationResult`, `OutputValidator`, `TurnHook`, `StructuredOutputConfig`, `StructuredRunResult`, and `StructuredAgentResult` types.
- Optional `telemetry` dependency group for OpenTelemetry packages.

### Changed

- **Runner** — Expanded `AgentRunner` with guardrail integration, structured output support, trace context propagation, and improved turn-level error handling. Significant internal refactor for extensibility.
- **Orchestrator** — `AnyCode` orchestrator now supports telemetry hooks, budget-aware scheduling, and structured task results. Task execution flow refactored for better observability.
- **Agent** — `Agent` class extended with guardrail config, trace context, and structured output options. Agent state management improved.
- **Scheduler** — Enhanced scheduling strategies with budget-aware task prioritization.
- **Providers** — `AnthropicAdapter` and `OpenAIAdapter` updated with streaming improvements, better error propagation, and structured output pass-through.
- **Tools** — `ToolRegistry` and `ToolExecutor` refined for safer execution, improved validation, and better error messages. `bash`, `file_write`, and `grep` tools hardened.
- **Collaboration** — `Team`, `MessageBus`, and `SharedMemory` refined with tighter type contracts and improved concurrency safety.
- **Types** — Expanded `types.py` with all new Pydantic models. All models remain frozen (immutable).

### Fixed

- Pool concurrency edge case in `AgentPool` under high parallelism.
- Task dependency validation now catches circular references earlier.

## [0.1.0] - 2025-03-20

### Added

- Initial release of the AnyCode Python orchestration framework.
- Core agent system with `Agent`, `AgentRunner`, `AgentPool`, and `Scheduler`.
- `AnyCode` high-level orchestrator with `TaskSpec` declarative API.
- Provider-agnostic LLM integration via `LLMAdapter` protocol with Anthropic and OpenAI adapters.
- Team collaboration primitives: `Team`, `MessageBus`, `SharedMemory`, `InMemoryStore`.
- Dependency-aware task scheduling with topological sort.
- Built-in tool system: `bash`, `file_read`, `file_edit`, `file_write`, `grep`.
- `ToolRegistry` with `define_tool` for runtime tool registration.
- `Semaphore`-based concurrency gating.
- Token usage tracking with `merge_usage`.
- Four examples: solo worker, crew workflow, staged pipeline, hybrid tooling.
- Pydantic-based immutable type system (`frozen=True` on all models).

[Unreleased]: https://github.com/Quantlix/anycode/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/Quantlix/anycode/compare/v0.8.2...v0.9.0
[0.8.2]: https://github.com/Quantlix/anycode/compare/v0.8.0...v0.8.2
[0.8.0]: https://github.com/Quantlix/anycode/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/Quantlix/anycode/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/Quantlix/anycode/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Quantlix/anycode/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Quantlix/anycode/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Quantlix/anycode/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Quantlix/anycode/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Quantlix/anycode/releases/tag/v0.1.0
