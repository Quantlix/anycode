---
title: "AnyCode — Multi-Agent AI Orchestration Framework for Python"
description: Build multi-agent AI workflows with AnyCode using typed tools, dependency-aware task graphs, provider-neutral models, durable runs, and verification gates.
keywords: python multi-agent framework, AI agent orchestration, LLM agent framework, agent task scheduling, tool use pydantic, multi-provider agents, agent verification
template: home.html
hide:
  - navigation
  - toc
---

<h1 class="ac-visually-hidden">AnyCode — multi-agent AI orchestration framework for Python</h1>

<section class="ac-section">
  <span class="ac-section__eyebrow">Why AnyCode</span>
  <h2 class="ac-section__title">Everything a team of agents needs, typed and observable</h2>
  <p class="ac-section__lede">One model call is rarely enough. AnyCode gives each agent a role, a model, scoped tools, task context, lifecycle events, and measurable results — so you can build workflows you can actually debug.</p>
</section>

<div class="ac-features">
  <a class="ac-card" href="concepts/agents-and-teams/">
    <span class="ac-card__icon" aria-hidden="true"><svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="9" cy="7" r="3"/><path d="M3 21v-1a6 6 0 0 1 6-6 6 6 0 0 1 6 6v1"/><path d="M16 3.5a3 3 0 0 1 0 7"/><path d="M21 21v-1a6 6 0 0 0-3-5.2"/></svg></span>
    <span class="ac-card__title">Coordinated agent teams</span>
    <span class="ac-card__body">Compose planner, builder, reviewer, and evaluator agents that share memory and pass messages through one runtime.</span>
  </a>
  <a class="ac-card" href="guides/multi-agent-team/">
    <span class="ac-card__icon" aria-hidden="true"><svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="5" cy="6" r="2.5"/><circle cx="19" cy="6" r="2.5"/><circle cx="12" cy="18" r="2.5"/><path d="M7 7.5l3.5 8M17 7.5l-3.5 8M7.5 6h9"/></svg></span>
    <span class="ac-card__title">Dependency-aware task graphs</span>
    <span class="ac-card__body">Declare <code>TaskSpec</code> dependencies and AnyCode topologically sorts them, then runs each ready task in concurrent waves.</span>
  </a>
  <a class="ac-card" href="guides/tools/">
    <span class="ac-card__icon" aria-hidden="true"><svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a4.5 4.5 0 0 0-6 5.6L3 17.6V21h3.4l5.7-5.7a4.5 4.5 0 0 0 5.6-6L14.5 12l-2.5-2.5 2.7-2.7z"/></svg></span>
    <span class="ac-card__title">Typed tools, validated inputs</span>
    <span class="ac-card__body">Ship built-in <code>bash</code>, file, and <code>grep</code> tools, or register your own with Pydantic input models and MCP servers.</span>
  </a>
  <a class="ac-card" href="concepts/overview/">
    <span class="ac-card__icon" aria-hidden="true"><svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 3v4a2 2 0 0 1-2 2H3M15 3v4a2 2 0 0 0 2 2h4M9 21v-4a2 2 0 0 0-2-2H3M15 21v-4a2 2 0 0 1 2-2h4"/></svg></span>
    <span class="ac-card__title">Provider-agnostic</span>
    <span class="ac-card__body">Mix Anthropic, OpenAI, Gemini, Ollama, Azure, and Bedrock in one team through a single typed <code>LLMAdapter</code> protocol.</span>
  </a>
  <a class="ac-card" href="guides/context-engineering/">
    <span class="ac-card__icon" aria-hidden="true"><svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a4 4 0 0 1 4 4c2 .5 3.5 2.2 3.5 4.4 0 1.7-.9 3.1-2.2 3.9M12 3a4 4 0 0 0-4 4c-2 .5-3.5 2.2-3.5 4.4 0 1.7.9 3.1 2.2 3.9M12 3v18M8 21h8"/></svg></span>
    <span class="ac-card__title">Memory &amp; context engineering</span>
    <span class="ac-card__body">Shared memory, RAG retrieval, checkpoints, and context policies that keep long runs inside the model's window.</span>
  </a>
  <a class="ac-card" href="guides/verification-gates/">
    <span class="ac-card__icon" aria-hidden="true"><svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l7 3v5c0 4.4-3 8.4-7 9.5C8 19.4 5 15.4 5 11V6l7-3z"/><path d="M9 11.5l2 2 4-4"/></svg></span>
    <span class="ac-card__title">Verification gates</span>
    <span class="ac-card__body">Run <code>ruff</code>, <code>pyright</code>, <code>pytest</code>, schema, and regex sensors as quality gates that can block a bad result.</span>
  </a>
  <a class="ac-card" href="guides/durability/">
    <span class="ac-card__icon" aria-hidden="true"><svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16v12H4z"/><path d="M8 7V4h8v3M8 12h8M8 16h5"/></svg></span>
    <span class="ac-card__title">Durable execution</span>
    <span class="ac-card__body">Checkpoint work, resume interrupted runs, and move persistence between in-memory, SQLite, and external backends.</span>
  </a>
  <a class="ac-card" href="guides/portable-infrastructure/">
    <span class="ac-card__icon" aria-hidden="true"><svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z"/><path d="M4.5 7.8L12 12l7.5-4.2M12 12v9"/></svg></span>
    <span class="ac-card__title">Portable operations</span>
    <span class="ac-card__body">Carry execution identity, policy, routing, telemetry, sandbox, and hosting contracts across deployment targets.</span>
  </a>
</div>

<section class="ac-section">
  <span class="ac-section__eyebrow">Bring your own model</span>
  <h2 class="ac-section__title">Works with the providers you already use</h2>
  <p class="ac-section__lede">Every provider sits behind the same protocol. Route work per task, or implement <code>LLMAdapter</code> for anything not listed.</p>
</section>

<div class="ac-providers">
  <span class="ac-chip">Anthropic</span>
  <span class="ac-chip">OpenAI</span>
  <span class="ac-chip">Google Gemini</span>
  <span class="ac-chip">Ollama</span>
  <span class="ac-chip">Azure OpenAI</span>
  <span class="ac-chip">AWS Bedrock</span>
  <span class="ac-chip">MCP tools</span>
  <span class="ac-chip">Custom adapters</span>
</div>

<section class="ac-section">
  <span class="ac-section__eyebrow">Get running in minutes</span>
  <h2 class="ac-section__title">Install, add a key, run a team</h2>
  <p class="ac-section__lede">Requires Python 3.12+. Only one provider key is needed for the basic examples.</p>
</section>

```bash
uv add anycode-py          # core
uv add "anycode-py[cli]"   # CLI and YAML/TOML configuration
```

```python title="one_agent.py"
import asyncio
from anycode import AnyCode

async def main() -> None:
    engine = AnyCode(config={"default_provider": "anthropic", "default_model": "claude-haiku-4-5"})
    result = await engine.run_agent(
        config={"name": "explainer", "system_prompt": "You explain Python clearly.", "tools": []},
        prompt="Explain what an async generator is in two sentences.",
    )
    print(result.output)

asyncio.run(main())
```

[Read the quickstart :material-arrow-right:](getting-started/quickstart.md){ .md-button .md-button--primary }
[Browse the concepts](concepts/overview.md){ .md-button }

<section class="ac-cta">
  <h2 class="ac-cta__title">Start orchestrating agent teams</h2>
  <p class="ac-cta__lede">Start with the Python runtime, or use the TypeScript service-client preview from Node.js and modern browsers.</p>
  <div class="ac-cta__actions">
    <a class="ac-btn ac-btn--inverse" href="getting-started/installation/">Install AnyCode</a>
    <a class="ac-btn ac-btn--ghost" href="https://github.com/Quantlix/anycode" target="_blank" rel="noopener">View on GitHub</a>
  </div>
</section>

!!! info "Agent-friendly by design"
    Coding agents can start from [`/llms.txt`](llms.txt) — a curated index of the most useful pages, examples, and source Markdown.
