---
title: "Semantic Contracts"
description: "Use AnyCode's versioned JSON contracts for runs, tasks, messages, artifacts, lifecycle state, event projections, fenced operations, and portability rules."
keywords: AnyCode semantic contract, JSON Schema, event projection, cancellation semantics, fencing token, artifact store
---

# Semantic Contracts

AnyCode ships a preview, language-neutral contract for describing runs and their durable evidence. It is a JSON data and behavior contract: it does not turn the embedded Python orchestrator into a network service or automatically replace the existing `Task`, `Message`, lifecycle-event, checkpoint, or tool-idempotency models.

The preview contract version is `1.0`. Its models are strict, frozen Pydantic models, and all extensible data fields use JSON values rather than Python objects. Writers include `schema_version`; future versions and unknown fields fail validation instead of being guessed.

## Domain model inventory

| Wire model | Purpose |
| --- | --- |
| `Run` | Run identity, state, generation, attempt, correlation, cancellation, and event cursor |
| `SemanticTask` | Task state, dependencies, attempts, and produced artifacts; exported as `Task` from `anycode.contracts` |
| `SemanticMessage` | Discriminated text, data, and artifact-reference parts; exported as `Message` from `anycode.contracts` |
| `Artifact` | Inline or referenced content with digest, provenance, classification, and retention |
| `Event` | Ordered run fact with payload version, sequence, correlation, causation, generation, and attempt |
| `Checkpoint` | Consistent run/task snapshot anchored to an event cursor and generation |
| `PolicyDecision` | Allow/deny outcome, policy version, reason codes, and obligations |
| `SemanticVerificationResult` | Verifier outcome and JSON evidence; exported as `VerificationResult` from `anycode.contracts` |
| `CapabilityDescriptor` | Implementation and contract versions, operations, artifact forms, and supported behaviors |

The aliases at the package root preserve the established runtime names. Import the unaliased semantic models from `anycode.contracts` when building a protocol adapter:

```python
from anycode.contracts import Message, RetryPolicy, Task, VerificationResult
```

Every recoverable contract failure is represented by `ContractError`. `Result` provides a JSON-only success/error envelope for adapter boundaries.

## State machine

Runs and semantic tasks share these states:

| Current | Legal next states |
| --- | --- |
| `accepted` | `queued`, `canceled`, `rejected` |
| `queued` | `running`, `waiting`, `failed`, `canceled` |
| `running` | `waiting`, `succeeded`, `failed`, `canceled` |
| `waiting` | `queued`, `running`, `failed`, `canceled` |
| `succeeded`, `failed`, `canceled`, `rejected` | none |

`waiting` always carries one typed reason: dependency, schedule, input, authorization, approval, retry backoff, capacity, or external signal. A transition to `failed` or `rejected` requires a typed error. Callers may provide an expected generation; a stale generation is rejected without producing an event.

Dependency evaluation is deterministic. Missing or active prerequisites wait; failed prerequisites fail the dependent; canceled prerequisites cancel it when every failed prerequisite was canceled. A task can explicitly opt into finalized partial artifacts from failed dependencies. That opt-in never converts an absent artifact into success.

## Cancellation and late results

Cancellation is a request/acknowledgement protocol rather than an instantaneous state mutation:

1. A non-terminal run records `cancellation.requested` and remains in its current state.
2. A worker acknowledgement records `cancellation.acknowledged` and settles the run as `canceled`.
3. Repeated requests and acknowledgements are idempotent and do not append duplicate events.
4. If success or failure is committed after the request but before acknowledgement, terminal completion wins and cancellation becomes `lost_to_completion`.
5. A request against an already terminal run also becomes `lost_to_completion`. A later result cannot transition a canceled or otherwise terminal run.

This ordering makes before-start, in-flight, and terminal races reproducible from the event history.

## Retry and resume

Only transient, rate-limited, and provider-unavailable classifications are retryable. The retry policy sets a maximum attempt count. Provider switching is separately opt-in and requires the candidate to share the current provider's compatibility class and, when configured, belong to the policy allowlist.

A checkpoint is valid only when its run identity, generation, embedded cursor, and schema version agree. A completed generation can start a new generation from a compatible checkpoint; an active generation cannot. The new generation resets state, attempt, cancellation, and terminal error while preserving the ordered event cursor.

## Events and projections

An event stream is one run's contiguous sequence beginning at one. Event IDs are unique, and `causation_id` may refer only to an earlier event in the stream. Invalid gaps, duplicates, mixed run IDs, and unknown causal references fail validation.

AnyCode includes two deliberately independent reference projections:

- `IncrementalRunProjector` applies one event at a time for streaming consumers.
- `project_run()` folds a validated history as a pure batch operation.

Contract tests replay the same immutable history through both implementations and require identical run state, generation, cursor, task states, artifacts, cancellation status, and applied event IDs.

## Fenced operations

`InMemoryOperationStore` is a reference claim store for AnyCode-committed side-effect results. A claim binds an operation key to a canonical input digest, owner, lease expiry, and monotonically increasing fencing token. Active duplicate owners receive `busy`; different input receives `conflict`; committed claims replay the same result artifact; an uncertain effect fails closed until reconciliation.

An expired claim can be acquired by a new owner with a higher fencing token. The old owner cannot commit after that point. These rules prevent duplicate committed results in the reference store, but they do not make an external API transactional and are not a general exactly-once guarantee. A multi-host deployment needs a durable shared implementation and must pass the fencing token to a downstream system capable of enforcing it.

## Artifacts

Artifacts carry a SHA-256 digest, media type, size, provenance, classification, retention, generation, attempt, and correlation fields. Content is a discriminated union:

- `inline` embeds UTF-8 or base64 data.
- `reference` stores a provider URI and optional expiry.

`LocalArtifactStore` is the preview filesystem implementation. It writes metadata and content atomically, uses content-addressed blobs for references, verifies size and digest on read, and rejects reuse of an artifact ID with different content. An application access hook can deny reads or writes; a denied read exposes neither bytes nor artifact metadata.

The hook is application policy, not host isolation. Filesystem permissions, encryption, identity, tenant boundaries, retention enforcement, malware scanning, and network policy remain deployment responsibilities.

## JSON Schemas

The nine checked-in schemas live under `src/anycode/contracts/schemas/v1/`. Regenerate them after an intentional model change and verify them in CI:

```bash
uv run python scripts/generate_contract_schemas.py
uv run python scripts/generate_contract_schemas.py --check
```

Schema changes require compatibility review, updated golden fixtures, and projection/state evidence. The preview remains additive within version `1.0`; a change that reinterprets or removes existing wire data requires a new contract version.

## Executable evidence

Run the complete credential-free flow:

```bash
uv run python examples/37_semantic_contract.py
```

The example transitions a run, acquires and commits a fenced operation, writes and verifies a referenced artifact, and checks that both projections reach the same five-event terminal view. Focused evidence lives in `tests/test_contract_models.py`, `tests/test_contract_state.py`, `tests/test_contract_projection.py`, `tests/test_contract_effects.py`, `tests/test_contract_artifacts.py`, and `tests/test_contract_schema.py`.
