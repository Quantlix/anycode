---
title: "Configure and Switch LLM Providers in AnyCode"
description: "Point AnyCode agents at Anthropic, OpenAI, Google, Ollama, Bedrock, or Azure — set env vars and extras, mix providers in one team, and tune retry resilience."
keywords: anycode providers, anthropic openai google ollama bedrock azure, create_adapter, provider api keys, ResilientAdapter, RetryPolicy, provider extras, multi-provider team
---

# Configure LLM Providers

AnyCode is provider-agnostic: every agent names a `provider` and a `model`, and the engine builds the right adapter for you behind a single interface. This guide shows the six built-in providers, the environment variables and install extras each needs, how to mix providers inside one team, and how the built-in retry-and-circuit-breaker resilience layer wraps every call.

## Supported providers

| `provider` | Models (examples) | Required env vars | Install extra |
| --- | --- | --- | --- |
| `anthropic` | `claude-haiku-4-5`, `claude-sonnet-5` | `ANTHROPIC_API_KEY` | `anycode-py[anthropic]` |
| `openai` | `gpt-4o-mini`, `gpt-5` | `OPENAI_API_KEY` | `anycode-py[openai]` |
| `google` | `gemini-2.5-flash` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | `anycode-py[google]` |
| `ollama` | any local model (e.g. `llama3.1`) | none (local server) | `anycode-py[ollama]` |
| `bedrock` | AWS-hosted model IDs | AWS creds + `AWS_DEFAULT_REGION` | `anycode-py[bedrock]` |
| `azure` | your deployment name | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` | `anycode-py[azure]` |

Every provider SDK is optional and lazy-loaded. If an extra is missing, the adapter raises an `ImportError` at construction with the exact `pip install "anycode-py[...]"` hint. Use `anycode-py[providers]` to install every built-in provider SDK.

!!! note "FakeAdapter for tests"
    `FakeAdapter` is a seventh, offline provider used for deterministic tests and CI. It returns scripted responses with zero network calls. See [Stream agent output](streaming.md) and [Run a multi-agent team](multi-agent-team.md) for how it slots in.

## Pick a provider per agent

The normal path is declarative: set `provider` and `model` on each `AgentConfig`. The engine resolves the adapter for you.

=== "Anthropic"

    ```python title="agent.py"
    from anycode import AgentConfig

    agent = AgentConfig(
        name="assistant",
        provider="anthropic",
        model="claude-haiku-4-5",
        tools=[],
    )
    ```

=== "OpenAI"

    ```python title="agent.py"
    from anycode import AgentConfig

    agent = AgentConfig(
        name="assistant",
        provider="openai",
        model="gpt-4o-mini",
        tools=[],
    )
    ```

=== "Ollama (local)"

    ```python title="agent.py"
    from anycode import AgentConfig

    agent = AgentConfig(
        name="assistant",
        provider="ollama",
        model="llama3.1",
        tools=[],
    )
    ```

!!! warning "provider defaults to Anthropic"
    If you leave `provider` unset (`None`), the agent falls back to `anthropic`. `model` is always required — there is no default model, so an agent with no `model` will not run.

## Mix providers in one team

Because each agent carries its own provider and model, a single team can spread work across vendors — a cheap, fast model for planning and a stronger model for review, for example.

```python title="mixed_team.py"
from anycode import AgentConfig, TeamConfig

team_config = TeamConfig(
    name="mixed-crew",
    shared_memory=True,
    agents=[
        AgentConfig(name="planner", provider="openai", model="gpt-4o-mini", tools=[]),
        AgentConfig(name="reviewer", provider="anthropic", model="claude-sonnet-5", tools=[]),
    ],
)
```

## Build an adapter directly

For scripts that call a model without the orchestrator, use the async `create_adapter` factory. It returns a ready adapter you can `chat()` or `stream()` against.

```python title="direct.py"
from anycode import create_adapter
from anycode.types import LLMChatOptions, LLMMessage, TextBlock

adapter = await create_adapter("openai")  # async — always await
messages = [LLMMessage(role="user", content=[TextBlock(text="Say hello in one sentence.")])]
result = await adapter.chat(messages, LLMChatOptions(model="gpt-4o-mini", max_tokens=64))
print(result.content[0].text, result.usage.input_tokens)
```

`create_adapter` takes provider-specific keyword arguments where a provider needs them:

| Keyword | Applies to | Purpose |
| --- | --- | --- |
| `api_key` | all keyed providers | Override the env-var key |
| `base_url` | ollama | Point at a non-default Ollama server |
| `endpoint`, `api_version` | azure | Azure resource endpoint and API version |
| `region`, `profile` | bedrock | AWS region and named profile |

## Resilience is on by default

Every adapter from `create_adapter` (and every agent run) is wrapped in a `ResilientAdapter`: a concurrency bulkhead, optional request pacing, automatic retries with exponential backoff and full jitter, a circuit breaker, and a per-call deadline. You get the default concurrency bound without configuring anything.

| Setting | Default | Meaning |
| --- | --- | --- |
| `RetryPolicy.max_attempts` | `6` | Attempts before giving up |
| `RetryPolicy.base_delay_seconds` | `1.0` | First backoff delay |
| `RetryPolicy.max_delay_seconds` | `60.0` | Backoff ceiling |
| `RetryPolicy.call_timeout_seconds` | `300.0` | Per-call deadline |
| `ProviderResilienceConfig.circuit_failure_threshold` | `5` | Consecutive failures before the circuit opens |
| `ProviderResilienceConfig.circuit_reset_seconds` | `120.0` | Wait before a half-open probe |
| `ProviderResilienceConfig.max_concurrency` | `8` | Simultaneous provider attempts per event loop and capacity scope; `None` disables |
| `ProviderResilienceConfig.requests_per_minute` | `None` | Evenly paced request starts; retries count; `None` disables |
| `ProviderResilienceConfig.capacity_scope` | provider name | Adapters that share this value share one limiter |
| `ProviderResilienceConfig.capacity_wait_timeout_seconds` | `300.0` | Queue wait before load shedding; `None` waits indefinitely |

The limiter is shared by adapters with the same scope and settings in one event loop, which is how agents in a normal team obey one local provider cap. Use distinct scopes for independent API-key quotas. Configuring different limits for one scope raises an error instead of silently creating a second pool. A stream holds its concurrency slot until it completes or is closed, and every retry reacquires both the concurrency and request-rate gates. When the queue wait expires, `ProviderCapacityError` (a `ProviderUnavailableError`) load-sheds without opening the provider circuit.

Retryable failures include timeouts, connection errors, and HTTP `408/409/429/500/502/503/504/529`; the layer honors a `Retry-After` header. Authentication and invalid-request errors are treated as terminal and never retried. When retries are exhausted or the circuit is open, `chat()` raises `ProviderUnavailableError`; a streaming call instead emits a terminal `error` event.

```python title="resilience.py"
from anycode import ProviderResilienceConfig, RetryPolicy, create_adapter

# Bound parallel calls, pace request starts, and tighten retries.
adapter = await create_adapter(
    "anthropic",
    resilience=ProviderResilienceConfig(
        max_concurrency=4,
        requests_per_minute=120,
        capacity_scope="anthropic-production-key",
        retry=RetryPolicy(max_attempts=3, call_timeout_seconds=60.0),
    ),
)

# Or opt out entirely for a bare adapter.
raw = await create_adapter("anthropic", resilience=ProviderResilienceConfig(enabled=False))
```

The request pacer controls request starts, not tokens per minute. Provider-side token quotas, organization-wide quotas, and traffic from other processes remain authoritative. In a multi-process or multi-host deployment, enforce the aggregate limit in your provider gateway or a distributed rate limiter as well as using this local bulkhead.

Set one policy for all agents in Python, then override it only where a separate quota is intentional:

```python title="team-capacity.py"
from anycode import AnyCode, OrchestratorConfig, ProviderResilienceConfig

engine = AnyCode(
    OrchestratorConfig(
        provider_resilience=ProviderResilienceConfig(
            max_concurrency=4,
            requests_per_minute=120,
            capacity_scope="shared-production-key",
        )
    )
)
```

!!! tip "Register your own provider"
    Need a vendor AnyCode doesn't ship? Register an async provider factory under a new name and it becomes usable everywhere `provider="..."` is accepted. See [Extend AnyCode with plugins](plugins.md).

## Next steps

- [Use reasoning models](reasoning-models.md) — reasoning effort and thinking budgets per provider.
- [Extend AnyCode with plugins](plugins.md) — register a custom provider factory.
- [Configuration reference](../reference/configuration.md) — every `AgentConfig` and resilience field.
- [Run a multi-agent team](multi-agent-team.md) — put mixed-provider agents to work.
