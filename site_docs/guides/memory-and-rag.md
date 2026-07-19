---
title: "Give AnyCode Agents Memory and RAG Retrieval"
description: "Give AnyCode agents persistent SQLite or Redis memory, team-shared context, and RAG retrieval so relevant prior knowledge is injected automatically at run time."
keywords: anycode memory, rag retrieval, create_memory_store, MemoryConfig, RAGConfig, shared memory, vector store, chromadb, CompositeMemory, KnowledgeStore
---

# Memory and RAG

By default an AnyCode agent remembers nothing between runs. This guide adds three layers on top: **key-value memory** (with optional sqlite/redis persistence), **shared memory** so a team's agents read each other's notes, and **RAG** so agents automatically retrieve relevant context from past work before they answer.

## Choose a memory backend

`create_memory_store` builds a key-value `MemoryStore` from a `MemoryConfig`. The default is in-memory (non-persistent); sqlite and redis survive restarts but need an extra installed.

| `backend` | Store | Persistence | Install extra |
| --- | --- | --- | --- |
| `"memory"` (default) | `InMemoryStore` | none | core |
| `"sqlite"` | `SQLiteStore` | file or `:memory:` | `anycode-py[persistence]` |
| `"redis"` | `RedisStore` | Redis server | `anycode-py[redis]` |

```python title="memory.py"
from anycode import create_memory_store
from anycode.types import MemoryConfig

store = create_memory_store(MemoryConfig(backend="sqlite", path="agent_memory.db"))
await store.setup()                       # required for sqlite and redis
await store.set("decision:auth", "Chose JWT with 15-minute expiry")
entry = await store.get("decision:auth")
await store.teardown()
```

`MemoryConfig.redact_sensitive_data` defaults to `True`. SQLite, Redis, and Chroma replace recognized credentials before writing values, metadata, or documents. `KnowledgeStore` applies the same default to its Markdown entries.

!!! warning "Redacted memory cannot reconstruct a credential"
    Retrieval returns `<redacted-secret>` rather than the original value. This is deliberate: long-lived memory should not be a credential store. Set `redact_sensitive_data=False` only when exact values are required and the backend has independent encryption, access control, and retention enforcement.

!!! warning "sqlite and redis need `setup()`"
    `SQLiteStore`, `RedisStore`, and `ChromaDBVectorStore` all raise a `RuntimeError` if you use them before `await setup()`. `InMemoryStore` needs no setup. Persistent stores are imported from their submodules (for example `from anycode.memory.sqlite_store import SQLiteStore`), not the top-level package.

## Share memory across a team

Set `shared_memory=True` on a `TeamConfig` and every agent gets a namespaced scratchpad each can read and write. Optionally inject a persistent `MemoryStore` so the shared notes survive the process.

```python title="shared.py"
from anycode import AgentConfig, TeamConfig

team_config = TeamConfig(
    name="research-crew",
    shared_memory=True,          # agents can read what earlier agents recorded
    agents=[
        AgentConfig(name="scout", provider="anthropic", model="claude-haiku-4-5", tools=[]),
        AgentConfig(name="writer", provider="anthropic", model="claude-haiku-4-5", tools=[]),
    ],
)
```

## Enable RAG retrieval

Retrieval-augmented generation lets an agent pull relevant snippets from a vector store before it answers. AnyCode wires this into the orchestrator: turn it on with a `RAGConfig`, and the engine indexes successful agent outputs and prepends the top matches to later prompts — automatically.

```python title="rag.py"
from anycode import AnyCode, OrchestratorConfig, RAGConfig
from anycode.types import MemoryConfig

config = OrchestratorConfig(
    memory=MemoryConfig(vector_backend="chromadb", vector_path="rag_index"),
    rag=RAGConfig(enabled=True, top_k=3, min_relevance=0.3, namespace="support"),
)
engine = AnyCode(config)
```

The knobs that shape retrieval:

| `RAGConfig` field | Default | Effect |
| --- | --- | --- |
| `enabled` | `False` | RAG is off until you set this |
| `auto_index` | `True` | Index successful agent outputs as they happen |
| `top_k` | `5` | Max snippets retrieved per query |
| `min_relevance` | `0.3` | Drop matches below this similarity score |
| `max_context_tokens` | `2000` | Token budget for injected context |
| `namespace` | `"default"` | Isolate one workload's memory from another |

The vector backend comes from `MemoryConfig.vector_backend`:

| `vector_backend` | Store | Notes |
| --- | --- | --- |
| `"none"` (default) | in-memory TF-IDF | **Not persistent** — logs a warning |
| `"memory"` | in-memory TF-IDF | Zero-dependency, non-persistent |
| `"chromadb"` | `ChromaDBVectorStore` | Persistent; needs `anycode-py[vector]` |

!!! danger "The default vector backend forgets everything"
    `MemoryConfig` defaults `vector_backend` to `"none"`, which uses a non-persistent in-memory index and emits a loud warning. For durable RAG you must set `vector_backend="chromadb"` and a `vector_path`. Also note `RAGConfig.enabled` defaults to `False` and `min_relevance=0.3` filters out weak matches — lower it if early testing returns nothing.

## Combine key-value and vector memory

`CompositeMemory` wraps a KV store and a vector store behind one object. With `auto_index=True`, every `set` also becomes searchable.

```python title="composite.py"
from anycode import CompositeMemory, InMemoryVectorStore, create_memory_store

memory = CompositeMemory(
    kv_store=create_memory_store(),
    vector_store=InMemoryVectorStore(),
    auto_index=True,
)
await memory.set("arch:events", "Chose an event-driven architecture for the ingest path")
hits = await memory.search("deployment and messaging", top_k=2)
```

## Give agents explicit memory tools

For a curated, human-readable knowledge base, `KnowledgeStore` writes one Markdown file per entry. `build_knowledge_tools` returns two tools an agent can call — `knowledge_save` to record a finding and `memory_search` to look one up.

```python title="knowledge.py"
from anycode import KnowledgeStore, build_knowledge_tools

store = KnowledgeStore(root="team_knowledge")
knowledge_tools = build_knowledge_tools(store)   # [knowledge_save, memory_search]
```

Register those tools like any other (see [Work with tools](tools.md)) and add `"knowledge_save"` / `"memory_search"` to an agent's `tools` allowlist.

For an independently protected knowledge directory that must preserve exact text, construct `KnowledgeStore(root="team_knowledge", redact_sensitive_data=False)` explicitly.

## The complete, runnable program

The snippets above each show one layer. Here is a complete `rag_memory.py` that ties the RAG layer together end to end: it enables retrieval with auto-indexing, runs one wave that records a fact, then a second wave whose answer can only come from retrieving what the first wave stored. It uses the zero-dependency in-memory vector backend, so no extra install is needed, and resolves a provider from whichever API key you have set.

```python title="rag_memory.py"
import asyncio
import os
import sys

from dotenv import load_dotenv

from anycode import AgentConfig, AnyCode, OrchestratorConfig, RAGConfig, TaskSpec, TeamConfig

load_dotenv()


def resolve_provider() -> tuple[str, str]:
    """Pick a provider and model from whichever API key is set."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", "claude-haiku-4-5"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", "gpt-4o-mini"
    sys.exit("Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your environment or .env file.")


PROVIDER, MODEL = resolve_provider()


async def main() -> None:
    # RAG on, auto-indexing every successful output. min_relevance=0.0 keeps this
    # demo from filtering out its single indexed fact; raise it for real workloads.
    config = OrchestratorConfig(
        rag=RAGConfig(enabled=True, auto_index=True, top_k=3, min_relevance=0.0, namespace="demo"),
    )
    engine = AnyCode(config)

    team = engine.create_team(
        "research",
        TeamConfig(
            name="research",
            agents=[
                AgentConfig(
                    name="scribe",
                    provider=PROVIDER,
                    model=MODEL,
                    system_prompt="You take careful notes and recall earlier facts when relevant.",
                    tools=[],
                ),
            ],
        ),
    )

    print(f"=== RAG Memory ({PROVIDER}, {MODEL}) ===\n")

    print("[Wave 1] Recording a memorable fact...")
    r1 = await engine.run_tasks(
        team,
        [
            TaskSpec(
                title="record-fact",
                description="State this exact fact in your own words: The capital of Atlantis is Coralis.",
                assignee="scribe",
            ),
        ],
    )
    print(f"  -> {r1.agent_results['scribe'].output[:140]}\n")

    print("[Wave 2] Asking a follow-up that must retrieve the fact...")
    r2 = await engine.run_tasks(
        team,
        [
            TaskSpec(
                title="recall-fact",
                description="Using only what you have learned in past sessions, what is the capital of Atlantis?",
                assignee="scribe",
            ),
        ],
    )
    print(f"  -> {r2.agent_results['scribe'].output[:240]}")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it from the project root:

```bash
uv run python rag_memory.py
```

!!! tip "Tested copy"
    See [`examples/15_rag_memory.py`](https://github.com/Quantlix/anycode/blob/main/examples/15_rag_memory.py) for the CI-tested version of this RAG walkthrough, and [`examples/06_pluggable_memory.py`](https://github.com/Quantlix/anycode/blob/main/examples/06_pluggable_memory.py) for every KV, vector, and composite memory backend exercised end to end.

## Next steps

- [Build a research assistant with memory](../tutorials/research-assistant.md) — a full RAG project end to end.
- [Engineer the context window](context-engineering.md) — control what fills the model's context as history grows.
- [Run a multi-agent team](multi-agent-team.md) — the team that shares this memory.
- [Configuration reference](../reference/configuration.md) — every `MemoryConfig` and `RAGConfig` field.
