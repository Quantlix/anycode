---
title: "Architecture Decision Record Template"
description: "Copyable AnyCode ADR template covering context, decision, state and failure semantics, compatibility, security, evidence, rollout, and rollback."
keywords: ADR template, AnyCode architecture template, design decision checklist
---

# ADR NNNN: Short Decision Title

- **Status:** proposed
- **Date:** YYYY-MM-DD
- **Owners:** names or team
- **Reviewers:** required maintainers
- **Supersedes:** ADR number or none
- **Superseded by:** ADR number or none

## Context

Describe the concrete workload, current behavior, constraint, and evidence that requires a decision. Name the public, persisted, protocol, security, or operational boundary affected.

## Decision drivers

- Driver with a measurable consequence.
- Compatibility or interoperability constraint.
- Failure, security, privacy, or operational requirement.

## Considered options

### Option A: Name

Describe behavior, benefits, costs, and rejected risks.

### Option B: Name

Describe behavior, benefits, costs, and rejected risks.

## Decision

State the selected option and the normative invariants. Define identifiers, versions, authoritative state, ownership, ordering, and terminal behavior where applicable.

## State and failure semantics

For every applicable case, state the authoritative outcome, emitted evidence, and repair path:

- validation or admission rejection;
- cancellation before start and while running;
- timeout, retry, and exhausted retry;
- duplicate delivery or stale ownership;
- process or backend failure before and after a side effect;
- partial artifact or checkpoint write;
- incompatible reader, writer, or peer version.

## Compatibility and migration

Name the oldest supported reader, additive and breaking changes, migration order, downgrade behavior, fixtures, and deprecation window. Never assign a new meaning to an existing persisted field.

## Security and privacy

Identify trust boundaries, authorization context, credential handling, data classification, redaction, retention, denial behavior, and operator-owned controls. State what the decision does not secure.

## Consequences

### Positive

- Concrete benefit.

### Negative

- Cost, limitation, or maintenance burden.

## Contract evidence

List golden fixtures, property tests, fault injection, independent implementations, and conformance commands. Evidence must fail before the implementation and pass after it.

## Rollout and rollback

Describe feature negotiation, deployment order, observability, stop conditions, rollback steps, and persisted-data handling. A rollback is incomplete if the old reader cannot read newly written data.

## Unresolved risks

List the owner and acceptance condition for each risk, or state `none`.
