---
title: "Architecture Decision Records"
description: "Write reviewable AnyCode architecture decisions with explicit contracts, failure semantics, compatibility, security boundaries, evidence, and rollback."
keywords: AnyCode ADR, architecture decision record, design review, compatibility decision
---

# Architecture Decision Records

Use an architecture decision record (ADR) when a change defines or revises a public API, persisted format, lifecycle transition, event meaning, side-effect rule, protocol mapping, security boundary, or backend contract. An ADR records the choice and its consequences; executable contract tests prove it.

Store accepted records in a tracked `decisions/` directory using `NNNN-short-title.md`. Do not reuse a number or rewrite an accepted decision in place. A later ADR supersedes it and links both directions.

## Required review

An ADR must identify:

- the observable contract and affected users or persisted data;
- alternatives considered, including keeping the current behavior;
- failure, cancellation, retry, duplicate-delivery, and recovery behavior where relevant;
- compatibility, migration, rollback, security, privacy, and operational consequences;
- conformance fixtures and tests that make the decision executable;
- unresolved risks with an owner or explicit deferral boundary.

Approval follows the [maintainer governance policy](maintainers.md). A draft can accompany an experiment, but a public or persisted contract does not become accepted on prose alone.

## Template

Copy the [ADR template](adr-template.md) and replace every instructional placeholder. Keep the decision narrow enough that one reviewer can explain its invariants and rollback path.

## Lifecycle

| Status | Meaning |
| --- | --- |
| `proposed` | Open for design and evidence review. |
| `accepted` | Approved and backed by the named contract tests. |
| `rejected` | Considered but not selected; rationale remains useful. |
| `superseded` | Replaced by a linked later ADR. |
| `deprecated` | Still readable for history but no longer recommended. |

Implementation and ADR status must agree. If evidence invalidates an accepted decision, write a superseding ADR rather than silently changing the contract.
