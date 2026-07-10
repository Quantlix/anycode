---
title: "AnyCode for TypeScript — Coming Soon"
description: A TypeScript SDK for AnyCode is in planning. Use the Python framework today, and follow along for the TypeScript agent-orchestration API.
keywords: anycode typescript, typescript agent framework, typescript multi-agent orchestration, node ai agents
hide:
  - toc
---

# AnyCode for TypeScript

<span class="ac-pill ac-pill--alpha">Planned</span>

A first-class **TypeScript SDK** for AnyCode is on the roadmap. The goal is the same typed, async-first orchestration model you get in Python — agents, dependency-aware task graphs, tools, memory, and verification gates — expressed idiomatically for Node and the browser-adjacent runtime.

!!! note "Not shipped yet"
    The TypeScript API does not exist on npm today. This page reserves the language switcher and lets you track progress. For anything you want to build **now**, use the [Python framework](../index.md) — it is complete and actively developed.

## What we are aiming for

The TypeScript SDK is being designed to mirror the Python surface so concepts transfer directly:

<div class="ac-features" markdown>

<span class="ac-card">
  <span class="ac-card__icon">🧭</span>
  <span class="ac-card__title">Same mental model</span>
  <span class="ac-card__body">Agents, teams, and <code>TaskSpec</code> dependencies map one-to-one with the Python API.</span>
</span>

<span class="ac-card">
  <span class="ac-card__icon">🔒</span>
  <span class="ac-card__title">End-to-end types</span>
  <span class="ac-card__body">Zod-validated tool inputs and fully typed run results, matching Python's Pydantic models.</span>
</span>

<span class="ac-card">
  <span class="ac-card__icon">🔌</span>
  <span class="ac-card__title">Provider parity</span>
  <span class="ac-card__body">The same provider adapters and a bring-your-own-adapter protocol.</span>
</span>

</div>

## A sketch of the intended API

This is a **design sketch**, not a released interface. Names and shapes may change.

```typescript
import { AnyCode } from "anycode";

const engine = new AnyCode({ maxConcurrency: 3 });

const team = engine.createTeam("crew", {
  sharedMemory: true,
  agents: [
    { name: "planner", provider: "anthropic", model: MODEL },
    { name: "builder", provider: "anthropic", model: MODEL },
    { name: "reviewer", provider: "anthropic", model: MODEL },
  ],
});

const result = await engine.runTasks(team, [
  { title: "Plan", assignee: "planner" },
  { title: "Build", assignee: "builder", dependsOn: ["Plan"] },
  { title: "Review", assignee: "reviewer", dependsOn: ["Build"] },
]);
```

## Follow along

- ⭐ Star and watch the [GitHub repository](https://github.com/Quantlix/anycode) for release news.
- 💬 Open a [discussion or issue](https://github.com/Quantlix/anycode/issues) to tell us what you need from the TypeScript SDK.
- 🐍 In the meantime, [start with the Python framework](../getting-started/quickstart.md).
