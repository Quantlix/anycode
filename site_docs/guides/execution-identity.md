---
title: "Propagate AnyCode Execution Identity and Policy"
description: Carry immutable principal, tenant, delegation, region, classification, and credential references through AnyCode and enforce external policy.
keywords: AnyCode execution context, AI agent identity, agent policy enforcement, tenant scoped agents, workload identity
---

# Propagate execution identity and enforce policy

`ExecutionContext` is AnyCode's immutable identity envelope for a run. It carries principal, tenant, delegation, data classification, placement, trace, and credential references across model calls, tools, durable work, external signals, sandbox requests, policy input, and telemetry. It stores references to credentials, never credential values.

## Create an execution context

```python
from anycode import DelegationGrant, ExecutionContext

context = ExecutionContext(
    principal="user:reviewer-42",
    subject="document:release-notes",
    workload_identity="kubernetes:serviceaccount:anycode-service",
    tenant_scope="tenant:example",
    delegation=(
        DelegationGrant(
            delegator="user:reviewer-42",
            delegatee="agent:release-reviewer",
            scopes=("document:read", "review:write"),
        ),
    ),
    classification="confidential",
    allowed_regions=("eu-west",),
    required_region="eu-west",
    credential_references=("vault:providers/reviewer",),
    trace_id="4f3c2a1b0e9d8c7b6a5f4e3d2c1b0a99",
)
```

The model rejects duplicate allowed regions, a required region outside the allowed set, malformed credential references, credential-like attribute keys, and common raw-secret patterns. A credential reference must contain a provider prefix such as `env:OPENAI_API_KEY` or `vault:providers/reviewer`.

## Attach identity to an agent

Pass the context through `AgentConfig`. The runner copies it into model options and tool context, adds bounded audit fields to spans, and uses its trace ID when one is provided.

```python
from anycode import AgentConfig

agent = AgentConfig(
    name="release-reviewer",
    provider="openai",
    model="configured-model",
    tools=[],
    execution_context=context,
)
```

`audit_attributes()` exposes principal, tenant, classification, and optional subject, workload identity, and region. Credential references are intentionally absent from that audit view.

## What belongs in `ExecutionContext`?

| Field | Purpose | Example |
| --- | --- | --- |
| `principal` | Identity accountable for the operation | `user:reviewer-42` |
| `subject` | Resource or user the work concerns | `document:release-notes` |
| `workload_identity` | Host-issued service identity reference | `kubernetes:serviceaccount:anycode-service` |
| `tenant_scope` | Isolation and policy scope | `tenant:example` |
| `delegation` | Explicit delegator, delegatee, scopes, and expiry | User delegates review scopes to an agent |
| `classification` | Data handling class | `public`, `internal`, `confidential`, or `restricted` |
| `allowed_regions` and `required_region` | Placement constraints | Require `eu-west` |
| `credential_references` | Provider-qualified lookup references | `vault:providers/reviewer` |
| `trace_id` | Correlation with an existing distributed trace | A 32-character trace identifier |
| `attributes` | Validated JSON policy input | Department or workload labels without secrets |

The host owns authentication, credential resolution, rotation, and revocation. `ExecutionContext` preserves the decision input and audit trail; it is not a secret store or an identity provider.

## Enforce an external policy decision

Implement an async adapter that returns the versioned `PolicyDecision` contract. The decision must match the run, task, correlation, generation, and attempt in the request.

```python
from anycode import PolicyDecision, PolicyEnforcer, PolicyRequest, uuid7


class TenantPolicy:
    async def decide(self, request: PolicyRequest) -> PolicyDecision:
        allowed = request.context.tenant_scope == "tenant:example"
        return PolicyDecision(
            id=str(uuid7()),
            run_id=request.run_id,
            task_id=request.task_id,
            outcome="allow" if allowed else "deny",
            policy_version="tenant-policy/1",
            reason_codes=("tenant_allowed" if allowed else "tenant_denied",),
            correlation_id=request.correlation_id,
            causation_id=request.causation_id,
            generation=request.generation,
            attempt=request.attempt,
        )


enforcer = PolicyEnforcer(TenantPolicy(), fail_closed=True)

result = await enforcer.enforce(
    PolicyRequest(
        run_id="run-42",
        task_id="task-review",
        action="model.invoke",
        resource="model:review-model",
        boundary="model",
        context=context,
        correlation_id="correlation-42",
    )
)

if not result.allowed:
    raise PermissionError(result.error.message if result.error else "denied")
```

When `fail_closed=True`, a missing adapter or adapter failure produces a deny decision. A mismatched decision context also denies the request. Set `fail_closed=False` only at a boundary where local allow-on-unavailability behavior is an explicit and reviewed policy.

## Handle policy obligations

An allow decision can include obligations, such as applying a data filter or recording a required approval. Register one async handler per obligation type. Every obligation must return `True`; an unknown, failed, or false obligation turns the enforcement result into `policy_obligation_unfulfilled`.

This ordering matters: policy is evaluated, obligations are fulfilled, and only then should the protected operation run. Do not execute the operation first and treat the policy result as an audit-only event.

## Understand failure and audit behavior

| Situation | Error code | Decision reason |
| --- | --- | --- |
| Adapter returns `deny` | `policy_denied` | Adapter-supplied reason |
| Required adapter is missing | `policy_denied` | `policy_required` |
| Adapter raises while fail-closed | `policy_denied` | `policy_unavailable` |
| Decision identifiers do not match the request | `policy_denied` | `invalid_policy_decision_context` |
| Obligation has no successful handler | `policy_obligation_unfulfilled` | The allow decision remains available for audit |
| Audit sink raises | No change | The decision remains effective; audit export failure does not corrupt run state |

The audit sink receives bounded identity attributes and obligation types. Keep the policy engine's own decision log as the authoritative policy audit record, and monitor export failures separately.

## Next steps

- [Route models with hard policy constraints](policy-routing.md)
- [Run commands through sandbox providers](sandbox-providers.md)
- [Map GenAI telemetry safely](genai-telemetry.md)
- [Review the security boundary](../reference/security.md)
