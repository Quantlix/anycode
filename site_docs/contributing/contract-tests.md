---
title: "Contract Test Conventions"
description: "Conventions for AnyCode lifecycle, persistence, protocol, backend, and provider contract suites with golden fixtures and fault injection."
keywords: AnyCode contract tests, conformance suite, golden fixtures, fault injection
---

# Contract Test Conventions

Contract tests define behavior that every implementation of a public protocol or persisted format must share. They are distinct from unit tests: a backend or adapter should be replaceable while the same suite and fixtures remain authoritative.

## Suite shape

Expose reusable suites as ordinary pytest functions or factory-generated tests. Pass an implementation factory or fixture into the suite; do not branch expected behavior by implementation name. Backend-specific guarantees belong in separate tests and documentation.

Every contract suite must cover:

- successful construction and the smallest valid operation;
- all legal states and representative illegal transitions;
- typed validation and recoverable error results;
- cancellation, deadline, retry, and late-result behavior;
- duplicate delivery, idempotency, and stale ownership where work can repeat;
- ordering, correlation, causation, and cursor behavior for events;
- cleanup, resource bounds, and failure isolation;
- version and capability reporting.

## Golden fixtures

Persisted and wire fixtures live under `tests/fixtures/compat/` in a directory named for the contract and version. A golden fixture is immutable evidence from a supported released writer. Add a new fixture for a new version; do not rewrite an old fixture to make a reader test pass.

For each supported version, test:

1. the current reader accepts the fixture;
2. projected semantic state matches explicit expectations;
3. absent additive fields receive documented defaults;
4. the next unsupported future version fails clearly;
5. migrations are deterministic and do not discard required evidence.

## State-machine properties

Generate every state pair from the public literal or enum. Assert all table-approved transitions succeed and every other transition fails without modifying state. Test terminal states explicitly. For event projections, replay the same fixture through at least two independently structured projectors when the contract requires implementation independence.

## Fault injection

Inject failure immediately before and after each durable or externally visible boundary. Name tests for the boundary and expected invariant, such as `test_stale_claim_cannot_commit_after_reassignment`. At minimum consider process exit, cancellation, timeout, duplicate delivery, storage conflict, partial write, unavailable policy, unavailable telemetry, and corrupted or future-version input.

Use a real child process when the guarantee concerns process death. Raising an exception in the same interpreter does not prove atomic filesystem visibility or cleanup-independent recovery.

## Determinism and isolation

- Use `FakeAdapter` or a protocol fake; unit and contract suites never call live providers.
- Use per-test temporary storage and release tasks, processes, environment changes, registries, clocks, and network resources.
- Freeze or assert ordering fields instead of relying on wall-clock sleeps.
- Keep semantic counts exact; apply timing thresholds only on controlled benchmark runners.
- Assert typed fields and events, not human-readable error text alone.

## Landing evidence

A contract change includes the ADR or compatibility rationale, suite changes, implementation changes, a focused command proving the new behavior, and the full repository gate. The pull request explains which old fixtures remain readable and which fault was injected.
