<p align="center">
  <img src="https://img.shields.io/pypi/v/anycode-py?style=flat-square&color=0078D4" alt="PyPI version" />
  <img src="https://img.shields.io/pypi/l/anycode-py?style=flat-square" alt="license" />
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/status-alpha-orange?style=flat-square" alt="Alpha status" />
  <img src="https://img.shields.io/badge/Built%20by-Quantlix-blueviolet?style=flat-square" alt="Built by Quantlix" />
</p>

# AnyCode

## Multi-agent AI orchestration framework for Python

> Developed and maintained by [Quantlix](https://github.com/Quantlix).

> **Alpha status and production boundary:** AnyCode is under active development. Supported top-level APIs and persisted formats have explicit compatibility contracts, but pre-1.0 minor releases may require migration. Production use is workload-specific: bounded deployments are eligible only after the [production readiness checklist](https://quantlix.github.io/anycode/latest/guides/production-readiness/) passes and the operator supplies host and network isolation, identity, durable storage, secrets management, monitoring, and incident controls. Direct safety-critical, critical-infrastructure, or unrestricted irreversible use is a no-go.

AnyCode is a Python framework for building coordinated AI agent teams. It helps developers compose autonomous LLM agents, connect them to typed tools, schedule dependent tasks, share memory, stream output, route work across providers, and inspect long-running agent workflows with explicit lifecycle and verification data.

If you are researching Python multi-agent orchestration, LLM agent frameworks, AI task scheduling, MCP tool integration, RAG memory, agent handoff, or DAG-based agent workflows, AnyCode is designed to give you a compact and strongly typed foundation to explore those patterns.

## What Is AnyCode?

AnyCode is an async-first orchestration layer for AI agents. A single agent can run a one-shot task, while a team can coordinate planning, implementation, review, memory, tool use, and validation through a shared runtime.

The framework focuses on practical harness engineering:

- Agent teams with shared memory and inter-agent messaging.
- Dependency-aware task execution using DAG scheduling and wavefront concurrency.
- Provider-agnostic LLM integration through a typed `LLMAdapter` protocol.
- Pydantic-validated tool calls and immutable runtime models.
- Observability, guardrails, structured output, checkpointing, HITL approval, MCP tools, routing, cost tracking, RAG memory, and verification gates.

AnyCode is built for experimentation, evaluation, local development, research prototypes, and bounded automation. A pinned deployment can be eligible for production when its workload-specific controls and evidence pass the readiness review; the package alone is not a production guarantee.

## Current Package

| Detail | Value |
| --- | --- |
| Distribution | `anycode-py` |
| Import package | `anycode` |
| Current version | `0.8.2` |
| Python | `>=3.12` |
| Project status | Alpha |
| License | MIT |
| Runtime style | Async-first |
| Core model style | Frozen Pydantic models |
| Build backend | Hatchling |

## Why Developers Use AnyCode

AnyCode is useful when you want more than a single chat loop. It gives each agent a role, a model, tool access, task context, lifecycle events, and measurable results.

Key use cases include:

- Build multi-agent AI workflows in Python.
- Run planner, builder, reviewer, and evaluator agents as one coordinated team.
- Execute task graphs with dependencies instead of manually sequencing prompts.
- Mix Anthropic, OpenAI, Google Gemini, Ollama, Azure OpenAI, and AWS Bedrock models.
- Register local tools or discover external tools through Model Context Protocol servers.
- Add validation layers such as structured output, content validators, cost budgets, approval gates, and quality sensors.
- Evaluate harness changes with deterministic fake adapters before using live LLM calls.

## Shipped Capabilities

| Area | What is available today |
| --- | --- |
| Core orchestration | `AnyCode`, `Agent`, `AgentRunner`, `AgentPool`, `Team`, `TaskQueue`, and `Scheduler` |
| Team coordination | `MessageBus`, `SharedMemory`, task queues, event callbacks, and team-level results |
| Task scheduling | Explicit `TaskSpec` dependencies, topological sort, wavefront execution, and cascading failure handling |
| Providers | Anthropic, OpenAI, Google Gemini, Ollama, AWS Bedrock, Azure OpenAI, plus custom `LLMAdapter` implementations |
| Tools | Built-in `bash`, `file_read`, `file_write`, `file_edit`, `grep`, and `list_files` tools, plus custom Pydantic tools |
| MCP | Connect to MCP servers, discover tools, register prefixed MCP tools, and scope MCP tools per agent |
| Safety and control | Guardrails, token and cost budgets, output validators, turn hooks, structured output, HITL approval gates |
| Persistence | In-memory, SQLite, Redis, vector memory, ChromaDB support, checkpoint stores, and resume support |
| [Portable infrastructure](https://quantlix.github.io/anycode/latest/guides/portable-infrastructure/) | Pluggable durability backends, execution identity, external policy enforcement, GenAI telemetry mapping, sandbox adapters, and hosting lifecycle contracts |
| Routing and handoff | Intelligent task routing, route decision reports, handoff requests, and context-preserving handoff execution |
| Advanced runtime | Cost reports, self-reflection, critic loops, DAG visualization, RAG memory, lifecycle states, stop reasons, and context engineering reports |
| Verification | Built-in `ruff`, `pyright`, `pytest`, `schema`, and `regex` sensors with quality gate decisions |
| Evaluation | Scenario loading, deterministic fake responses, benchmark reports, markdown rendering, and report comparison |
| Developer experience | CLI commands, YAML/TOML config, examples cookbook, CLI inspection, and deterministic eval reports |
| Extension ecosystem | Typed `Plugin` bundles (tools, provider factories, sensors, hooks) registered via `engine.register_plugin()` or auto-discovered through the `anycode.plugins` entry-point group |
| Service client | Dependency-free TypeScript preview for lifecycle, artifact, cancellation, and resumable-stream operations in Node.js 20+ and modern browsers |

Operational guides cover [durability backends](https://quantlix.github.io/anycode/latest/guides/durability-backends/), [execution identity and policy](https://quantlix.github.io/anycode/latest/guides/execution-identity/), [policy-constrained model routing](https://quantlix.github.io/anycode/latest/guides/policy-routing/), [sandbox providers](https://quantlix.github.io/anycode/latest/guides/sandbox-providers/), [service hosting](https://quantlix.github.io/anycode/latest/guides/hosting-services/), and [GenAI telemetry](https://quantlix.github.io/anycode/latest/guides/genai-telemetry/).

## Architecture At A Glance

```text
AnyCode orchestrator
  -> Team coordination
     -> AgentPool with bounded concurrency
     -> TaskQueue with dependency-aware scheduling
     -> MessageBus and SharedMemory
  -> AgentRunner
     -> LLMAdapter protocol
     -> ToolExecutor and ToolRegistry
     -> Guardrails, structured output, lifecycle, context policy, verification gates
  -> Optional systems
     -> Checkpointing, approval, MCP, routing, cost, reflection, RAG, evaluation
```

The main design rule is simple: the framework owns the harness, while providers and tools stay replaceable. Models are typed, immutable, and validated at runtime boundaries.

## Getting Started Guide

### 1. Requirements

- Python `3.12` or newer.
- `uv` for dependency management.
- At least one LLM API key for live examples.

### 2. Install AnyCode

For a new or existing Python project:

```bash
uv add "anycode-py[anthropic]"
```

For CLI and YAML/TOML configuration support:

```bash
uv add "anycode-py[cli]"
```

For the full optional ecosystem:

```bash
uv add "anycode-py[all]"
```

### 3. Add API Keys

Create a local `.env` file or export environment variables in your shell.

```bash
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key
GOOGLE_API_KEY=your-google-key
```

Only one supported provider is required to run the basic examples. Never commit API keys.

### 4. Run One Agent

```python
import asyncio
import os

from dotenv import load_dotenv

from anycode import AnyCode

load_dotenv()


def resolve_model() -> tuple[str, str]:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", "claude-haiku-4-5"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", "gpt-4o-mini"
    raise RuntimeError("Set ANTHROPIC_API_KEY or OPENAI_API_KEY first.")


async def main() -> None:
    provider, model = resolve_model()
    engine = AnyCode(config={"default_provider": provider, "default_model": model})

    result = await engine.run_agent(
        config={
            "name": "explainer",
            "provider": provider,
            "model": model,
            "system_prompt": "You explain Python clearly and briefly.",
            "tools": [],
            "max_turns": 2,
        },
        prompt="Explain what an async generator is in two sentences.",
    )

    print(result.output)
    print(f"tokens: in={result.token_usage.input_tokens} out={result.token_usage.output_tokens}")


asyncio.run(main())
```

### 5. Run A Team With Dependencies

```python
import asyncio
import os

from dotenv import load_dotenv

from anycode import AgentConfig, AnyCode, TaskSpec, TeamConfig

load_dotenv()

PROVIDER = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai"
MODEL = "claude-haiku-4-5" if PROVIDER == "anthropic" else "gpt-4o-mini"


async def main() -> None:
    engine = AnyCode(config={"max_concurrency": 3})

    team = engine.create_team(
        "guide-crew",
        TeamConfig(
            name="guide-crew",
            shared_memory=True,
            agents=[
                AgentConfig(
                    name="planner",
                    provider=PROVIDER,
                    model=MODEL,
                    system_prompt="Create concise technical plans.",
                    tools=[],
                ),
                AgentConfig(
                    name="writer",
                    provider=PROVIDER,
                    model=MODEL,
                    system_prompt="Turn plans into clear developer documentation.",
                    tools=[],
                ),
                AgentConfig(
                    name="reviewer",
                    provider=PROVIDER,
                    model=MODEL,
                    system_prompt="Review documentation for clarity and missing steps.",
                    tools=[],
                ),
            ],
        ),
    )

    result = await engine.run_tasks(
        team,
        [
            TaskSpec(
                title="Plan guide",
                description="Outline a getting started guide for a Python agent framework.",
                assignee="planner",
            ),
            TaskSpec(
                title="Draft guide",
                description="Write the guide using the plan from the planner.",
                assignee="writer",
                depends_on=["Plan guide"],
            ),
            TaskSpec(
                title="Review guide",
                description="Review the draft and list concrete improvements.",
                assignee="reviewer",
                depends_on=["Draft guide"],
            ),
        ],
    )

    print(f"success={result.success}")
    for agent_name, agent_result in result.agent_results.items():
        print(f"\n[{agent_name}]\n{agent_result.output[:600]}")


asyncio.run(main())
```

### 6. Use YAML Or TOML Config

Install the CLI extra first:

```bash
uv add "anycode-py[cli]"
```

Create `team.yaml`:

```yaml
name: guide-crew
shared_memory: true
max_concurrency: 3

agents:
  - name: planner
    provider: anthropic
    model: claude-haiku-4-5
    system_prompt: Create concise technical plans.
    tools: []

  - name: writer
    provider: anthropic
    model: claude-haiku-4-5
    system_prompt: Write clear developer documentation.
    tools: []

tasks:
  - title: Plan guide
    description: Outline a getting started guide for AnyCode.
    assignee: planner

  - title: Draft guide
    description: Write the guide from the plan.
    assignee: writer
    depends_on:
      - Plan guide

verification:
  - name: regex
    kind: computational
    phases:
      - after_team
    block_on_failure: true
    options:
      pattern: AnyCode
      expect: match
```

Run it:

```bash
uv run anycode run team.yaml
```

Or load the same config in Python:

```python
import asyncio

from anycode import AnyCode


async def main() -> None:
    engine = AnyCode.from_config("team.yaml")
    result = await engine.run_team_from_config()
    print(result.model_dump_json(indent=2, exclude_none=True))


asyncio.run(main())
```

## CLI Reference

The CLI is available through the `cli` extra.

```bash
uv run anycode init my-agent-project
uv run anycode run team.yaml
uv run anycode run --agent helper --provider anthropic --model claude-haiku-4-5 --prompt "Summarize async Python."
uv run anycode inspect tools
uv run anycode inspect providers
uv run anycode inspect team team.yaml
uv run anycode inspect config team.yaml
uv run anycode eval run tests/fixtures/eval/runtime_reliability_deterministic.yaml --variant baseline --markdown
uv run anycode eval compare artifacts/eval/baseline.json artifacts/eval/candidate.json
uv run anycode version
```

`anycode init` creates a small project with `team.yaml`, `main.py`, `.env.example`, a `tools/` package, and `.gitignore`.

## Providers And Optional Extras

| Extra | Purpose |
| --- | --- |
| `cli` | `anycode` CLI, Rich output, YAML parsing |
| `telemetry` | OpenTelemetry tracing and exporters |
| `persistence` | SQLite memory and checkpoint support |
| `redis` | Redis memory backend |
| `vector` | ChromaDB vector memory backend |
| `google` | Google Gemini adapter |
| `ollama` | Local Ollama adapter over HTTP |
| `bedrock` | AWS Bedrock adapter |
| `azure` | Azure OpenAI adapter |
| `mcp` | Model Context Protocol client and tool discovery |
| `sandbox` | Daytona sandbox adapter |

Provider support is protocol-based. You can bring your own adapter by implementing the `LLMAdapter` interface.

## Built-In Tools

| Tool | Purpose |
| --- | --- |
| `bash` | Execute shell commands with timeout and captured output |
| `file_read` | Read file contents with line and size controls |
| `file_write` | Create or overwrite files and parent directories |
| `file_edit` | Replace targeted text in existing files |
| `grep` | Search files using regex, with ripgrep when available |
| `list_files` | List project files while respecting repository ignore rules |

Custom tools use Pydantic input models and are registered through `define_tool()` and `ToolRegistry`.

## Examples Cookbook

The `examples/` directory contains 40 runnable scripts. They are arranged from beginner workflows to runtime reliability demos.

| Examples | Theme |
| --- | --- |
| `01_solo_worker.py` to `04_hybrid_tooling.py` | Single agents, teams, dependency pipelines, custom and built-in tools |
| `05_production_features.py` | Telemetry, guardrails, and structured output |
| `06_pluggable_memory.py` to `08_hitl_approval.py` | Memory stores, checkpointing, and human approval |
| `09_multi_provider.py` to `12_intelligent_routing.py` | Provider mixing, MCP tools, handoff, and routing |
| `13_cost_tracking.py` to `17_yaml_config.py` | Cost reports, reflection, RAG memory, DAG visualization, and YAML config |
| `18_execution_lifecycle.py` to `21_eval_suite.py` | Lifecycle events, adaptive context, quality gates, and evaluation suites |
| `22_deterministic_eval.py` to `25_runtime_cancellation.py` | Fake adapters, context pressure, verification gates, and cancellation telemetry |
| `26_context_engineering.py` | Huge-context model profiles, section budgets, first-class section inputs, and usage reports |
| `27_plugin_ecosystem.py` | Plugin bundles wiring custom tools, provider factories, and sensors into the engine |
| `28_durable_runs.py` to `30_scheduled_wakeups.py` | Durable resumable runs, session chaining, and scheduled wakeups |
| `31_streaming_runtime.py` to `34_list_files.py` | Provider-token streaming, reasoning-model controls, authenticated MCP over HTTP, and fast file listing |
| `35_lifecycle_contract.py` to `36_runtime_baseline.py` | Complete team verification evidence and reproducible local-runtime baselines |
| `37_semantic_contract.py` | Versioned semantic events, fenced operations, artifact integrity, and independent projections |
| `38_pluggable_durability.py` to `39_backend_failure_soak.py` | Backend portability, migrations, leases, fencing, and failure-soak behavior |
| `40_operational_portability.py` | Execution identity, policy enforcement, model routing, and GenAI telemetry mapping |

Run an example from the repository root:

```bash
uv run python examples/01_solo_worker.py
```

Most live examples require `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`. Deterministic evaluation examples can run without live LLM credentials.

## Development Setup

Clone the repository and install dependencies with `uv`:

```bash
git clone https://github.com/Quantlix/anycode.git
cd anycode
uv sync --locked --group dev
```

Run the local verification commands:

```bash
uv run python scripts/check_versions.py
uv run python -m ruff check .
uv run python -m ruff format --check src/
uv run python -m pyright
uv run python -m pytest
uv run python -m mkdocs build --strict
uv run python scripts/check_docs.py
```

The default pytest configuration excludes integration tests. Integration tests may require Docker services or provider credentials.

## Security And Responsible Use

AnyCode agents can be connected to tools that read files, write files, execute commands, call providers, and access external systems through MCP. Treat every tool-enabled agent as a privileged automation process.

Required safeguards for any deployment:

- Run tool-enabled agents as a non-root identity inside a container, VM, or equivalent isolation boundary.
- Set explicit agent tool lists and apply `ToolSecurityPolicy` with narrow path, shell, and environment access.
- Use fake adapters or deterministic evaluation before live model calls.
- Inject API keys from a protected secret source and keep them out of prompts, source, images, and logs.
- Use durable idempotency plus human approval for sensitive or irreversible actions.
- Allowlist production plugins and MCP endpoints, then enforce network egress outside the framework.
- Review the [security and threat model](https://quantlix.github.io/anycode/latest/reference/security/) and complete the [production readiness checklist](https://quantlix.github.io/anycode/latest/guides/production-readiness/).

Report suspected vulnerabilities through the private process in [SECURITY.md](SECURITY.md). Do not disclose vulnerability details in a public issue.

## FAQ

### Is AnyCode production ready?

Production readiness is workload-specific. AnyCode remains alpha, but a pinned release can support bounded, reversible, operator-monitored workloads when every mandatory readiness control passes. Customer-facing, multi-tenant, sensitive-data, or irreversible workflows require stronger application-owned isolation, authorization, storage, and operations. Direct safety-critical control, critical infrastructure control, and autonomous high-impact decisions are no-go uses for AnyCode as the sole decision or control system.

### What makes AnyCode different from a single-agent loop?

AnyCode gives you a team runtime: multiple agents, explicit task dependencies, shared memory, inter-agent messaging, scheduling strategies, provider routing, and structured run results.

### Can I use different LLM providers in one workflow?

Yes. Agents can use different providers and models in the same team. Routing can also select a model per task based on task complexity and configured rules.

### Can AnyCode run without live API keys?

Some examples require live providers. The deterministic evaluation suite and `FakeAdapter` support local tests without live LLM credentials.

### Does AnyCode support custom tools?

Yes. Tools are defined with Pydantic input models, registered at runtime, and executed through the same validation path as built-in tools.

## Contributing

Issues, discussions, and pull requests are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup, branch naming, compatibility review, tests, documentation, and pull request requirements. Repository collaborators use [MAINTAINERS.md](MAINTAINERS.md) for governance, change approval, backports, and release ownership.

## License

AnyCode is released under the MIT License. See [LICENSE](LICENSE) for details.

<p align="center">
  Built with purpose by <strong><a href="https://github.com/Quantlix">Quantlix</a></strong>
</p>
