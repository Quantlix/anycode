---
title: "Install AnyCode — Setup for the Python Multi-Agent Framework"
description: Install AnyCode with uv or pip on Python 3.12+, add optional extras for providers and tooling, configure your API keys, and verify the setup in minutes.
keywords: install anycode, anycode-py, pip install anycode, uv add anycode, python agent framework setup, anycode extras, provider api keys
---

# Install AnyCode

Install AnyCode in any Python 3.12+ project with a single command — `uv add anycode-py` or `pip install anycode-py`. This page covers that install, the optional extras for providers and tooling, provider key setup, and a quick check that the package imports correctly.

AnyCode ships as the `anycode-py` distribution on PyPI and imports as the `anycode` package. The core install stays deliberately light: provider SDKs, persistence backends, the CLI, and MCP support are opt-in [extras](#optional-extras) you add only when a workflow needs them.

!!! info "Alpha software"
    AnyCode is an alpha-stage framework under active development. APIs, defaults, and configuration formats can change between releases. It is well suited to prototypes, evaluation harnesses, and research — not yet to production workloads.

## Requirements

| Requirement | Details |
| --- | --- |
| Python | 3.12 or newer |
| Package manager | [`uv`](https://docs.astral.sh/uv/) recommended; `pip` also works |
| Provider key | At least one LLM API key for live runs (Anthropic, OpenAI, and others) |

The deterministic evaluation examples run without any provider key, so you can explore the framework offline before wiring up a live model.

## Install the AnyCode package

Add the core package to a new or existing project:

=== "uv"

    ```bash
    uv add anycode-py
    ```

=== "pip"

    ```bash
    pip install anycode-py
    ```

Add the `cli` extra when you want to scaffold and run workflows from YAML/TOML config files, with rich terminal output:

=== "uv"

    ```bash
    uv add "anycode-py[cli]"
    ```

=== "pip"

    ```bash
    pip install "anycode-py[cli]"
    ```

!!! tip "uv is recommended"
    Examples throughout these docs use `uv run …`, which resolves and runs commands against your project environment without a manual activation step. If you prefer `pip`, drop the `uv run` prefix and run the same commands inside your virtual environment.

## Optional extras

Extras are additive — combine any of them in a single bracketed list, and install only what a given workflow needs to keep your dependency tree small.

| Extra | Adds |
| --- | --- |
| `cli` | `anycode` command, YAML/TOML config, rich terminal output |
| `telemetry` | OpenTelemetry tracing support |
| `persistence` | SQLite-backed persistence helpers |
| `redis` | Redis memory support |
| `vector` | ChromaDB vector memory support |
| `mcp` | Model Context Protocol tools |
| `google` | Google Gemini provider support |
| `ollama` | Ollama provider support |
| `bedrock` | AWS Bedrock provider support |
| `azure` | Azure OpenAI provider support |
| `tokens` | Token counting through `tiktoken` |

!!! note "Anthropic and OpenAI need no extra"
    The Anthropic and OpenAI providers work with the core install. Add a provider extra only for Google Gemini (`google`), Ollama (`ollama`), AWS Bedrock (`bedrock`), or Azure OpenAI (`azure`).

For a broad local playground with every optional feature enabled:

=== "uv"

    ```bash
    uv add "anycode-py[cli,telemetry,persistence,redis,vector,mcp,google,ollama,bedrock,azure,tokens]"
    ```

=== "pip"

    ```bash
    pip install "anycode-py[cli,telemetry,persistence,redis,vector,mcp,google,ollama,bedrock,azure,tokens]"
    ```

## Configure provider keys

AnyCode reads provider credentials from your process environment. A local `.env` file works well for examples — the quickstart scripts load it with `load_dotenv()`:

```bash
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key
GOOGLE_API_KEY=your-google-key
```

Only one supported provider key is required for the basic examples — set whichever you have.

!!! warning "Keep secrets local"
    Do not commit `.env` files or API keys. Pass keys through environment variables, secret stores, or CI secrets.

## Verify the install

Confirm the package imports:

=== "uv"

    ```bash
    uv run python -c "from anycode import AnyCode; print(AnyCode.__name__)"
    ```

=== "pip"

    ```bash
    python -c "from anycode import AnyCode; print(AnyCode.__name__)"
    ```

The command should print:

```text
AnyCode
```

If you installed the `cli` extra, verify the command too:

=== "uv"

    ```bash
    uv run anycode version
    ```

=== "pip"

    ```bash
    anycode version
    ```

## Develop the AnyCode repository

Working inside the AnyCode repository itself? Install all development dependencies:

```bash
uv sync --group dev
```

Serve the documentation site locally with live reload:

```bash
uv run python -m mkdocs serve
```

Run the strict docs build used by CI:

```bash
uv run python -m mkdocs build --strict
```

## Next steps

- [Quickstart](quickstart.md) — run your first agent and a dependency-aware team.
- [Concepts overview](../concepts/overview.md) — how the orchestrator, agents, tasks, and tools fit together.
- [Work with tools](../guides/tools.md) — give agents built-in and custom Pydantic tools.
- [Public API reference](../reference/public-api.md) — the classes and functions you will import.
