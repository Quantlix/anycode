# Phase 2: Persistence & Resilience

> **Priority:** High
> **Status:** Complete
> **Goal:** Make workflows survive crashes and scale beyond memory

## Overview

Phase 2 addresses the fundamental limitation that all AnyCode state is in-memory and lost on restart. This phase introduces pluggable persistent storage, workflow checkpointing with resume, and human-in-the-loop approval gates.

## Features

| ID | Feature | Priority | Complexity | Status |
|----|---------|----------|------------|--------|
| 2.1 | [Pluggable Memory Backend](001-pluggable-memory.md) | High | Medium | Complete |
| 2.2 | [Workflow Checkpointing & Resume](002-checkpointing.md) | High | High | Complete |
| 2.3 | [Human-in-the-Loop Approval](003-human-in-the-loop.md) | Medium | Medium | Complete |

## Dependencies

- Feature 2.2 (checkpointing) benefits from 2.1 (persistent storage) but can use filesystem fallback
- Feature 2.3 is independent of 2.1 and 2.2

## New Dependencies

| Package | Purpose | Install Group |
|---------|---------|---------------|
| `aiosqlite` | Async SQLite for persistent KV store | `extras = ["persistence"]` |
| `redis[hiredis]` | Optional Redis backend | `extras = ["redis"]` |
| `chromadb` | Optional vector store | `extras = ["vector"]` |

## Test Coverage Target

85%+ for all Phase 2 modules.

## Acceptance Criteria

- [ ] Agent memory persists across process restarts (SQLite backend)
- [ ] Vector store supports semantic search over agent memories
- [ ] Custom memory backends work via the `MemoryStore` Protocol
- [ ] A 50-task workflow saves checkpoints after each wave
- [ ] A crashed workflow resumes from the last completed wave
- [ ] Checkpoint data is serializable and deserializable without data loss
- [ ] Human approval gate pauses execution and waits for input
- [ ] Approval timeout produces a configurable default action
- [ ] Approval works via callback, stdin, and webhook modes
