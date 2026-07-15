---
title: "Deploy Portable AnyCode Infrastructure"
description: Configure identity, external policy, sandbox providers, durable state, telemetry, and safe container or Kubernetes hosting for AnyCode.
keywords: AnyCode deployment, workload identity, Dapr state, Daytona sandbox, Kubernetes agents, external policy
---

# Deploy portable AnyCode infrastructure

The infrastructure preview separates semantic runtime behavior from provider and host controls. Your application chooses a `DurabilityBackend`, propagates an immutable `ExecutionContext`, enforces external policy where required, selects a `SandboxProvider`, and maps telemetry without putting credentials into run state.

## Propagate identity references

```python
from anycode import ExecutionContext

context = ExecutionContext(
    principal="user:reviewer-42",
    subject="document:release-notes",
    workload_identity="kubernetes:serviceaccount:anycode-service",
    tenant_scope="tenant:example",
    classification="confidential",
    allowed_regions=("eu-west",),
    required_region="eu-west",
    credential_references=("vault:providers/reviewer",),
)
```

Execution context reaches model and tool calls, durable work and signals, sandbox requests, policy input, and telemetry audit attributes. Values in `credential_references` are provider-qualified lookups; raw secrets and credential-like attributes are rejected. The host resolves references immediately before use and owns token lifetime and revocation.

Use `PolicyEnforcer(fail_closed=True)` when a policy decision is mandatory. Denials and obligations are typed, audit records contain only bounded identity metadata, and an unfulfilled obligation denies the action. Policy outages do not corrupt durable state.

## Select a sandbox provider

`CompanionSandboxAdapter` binds a host-supplied sandbox service to the stable provider protocol. `DaytonaSandboxProvider` is available through `anycode-py[sandbox]`; it supports lifecycle, commands, files, streaming, cancellation, evidence digests, and cleanup through the maintained async SDK. Its capability report marks stable snapshots unsupported, so callers cannot accidentally depend on a provider feature the adapter cannot prove.

`PolicySandboxProvider` applies external policy before sandbox actions. A provider capability report is descriptive, not an isolation certification. The provider and host remain responsible for kernel or VM isolation, egress enforcement, image provenance, tenant separation, storage erasure, and secret injection.

## Use the reference host profiles

The repository provides two profiles under `deploy/`:

| Profile | Intended boundary | State |
| --- | --- | --- |
| Generic container | One replica with a persistent volume and SQLite | Local operational profile |
| Kubernetes with Dapr | Rolling multi-replica host with service-account identity and external state | Integration reference |

Both configure non-root execution, immutable image references, liveness/readiness, graceful drain, resource limits, state, telemetry, and identity references. Agent Cards are generated at runtime from each public endpoint; do not bake production or canary URLs into the image.

### Host guarantees and AnyCode guarantees

| Host or provider owns | AnyCode owns |
| --- | --- |
| Placement, TLS, ingress authentication, workload identity, secret delivery, network policy, volumes, process signals, and termination timing | Semantic run/event models, backend claims and fencing, cancellation state, stream cursors, policy requests, sandbox protocol, capture policy, and capability reports |
| Dapr component correctness, state-store limits, backups, encryption, and regional placement | Optimistic concurrency use, version checks, state export, and backend-specific limitations |
| Sandbox isolation and egress enforcement | Provider-neutral commands, files, cancellation, evidence, cleanup, and explicit unsupported capabilities |

Before an upgrade, verify the candidate can read the deployed contract/backend versions and run conformance against a canary. Stop admission before drain; durably return work that misses the termination window. Roll back the immutable image without deleting or reversing durable history. Do not overlap versions whose supported contract ranges do not intersect.

See `deploy/IMAGE_CONTRACT.md` and `deploy/README.md` for the concrete image, upgrade, and rollback contract. `examples/40_operational_portability.py` runs identity, policy, telemetry, routing, and companion-sandbox behavior without credentials.
