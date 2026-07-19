---
title: "Deploy Portable AnyCode Infrastructure"
description: Deploy AnyCode portably with durable state, execution identity, external policy, sandbox providers, GenAI telemetry, and safe container or Kubernetes hosting.
keywords: AnyCode deployment, workload identity, Dapr state, Daytona sandbox, Kubernetes agents, external policy
---

# Deploy portable AnyCode infrastructure

The infrastructure preview separates semantic runtime behavior from provider and host controls. Your application chooses a `DurabilityBackend`, propagates an immutable `ExecutionContext`, enforces external policy where required, selects a `SandboxProvider`, and maps telemetry without putting credentials into run state.

## Choose the guide for your task

| Task | Guide |
| --- | --- |
| Store runs, events, claims, checkpoints, wakes, signals, and artifact references | [Configure durability backends](durability-backends.md) |
| Carry principal, tenant, delegation, classification, placement, and credential references | [Propagate execution identity and policy](execution-identity.md) |
| Filter models by classification, region, capability, budget, latency, and compatibility | [Route models with policy](policy-routing.md) |
| Run commands, streams, and files through Daytona or a companion service | [Run work through sandbox providers](sandbox-providers.md) |
| Add admission, readiness, graceful drain, durable return, and A2A discovery | [Host AnyCode services](hosting-services.md) |
| Map runtime events with capture policy, redaction, and failure-isolated export | [Map GenAI telemetry safely](genai-telemetry.md) |

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

See [Propagate execution identity and policy](execution-identity.md) for delegation, model and tool propagation, external adapters, obligations, and failure behavior.

## Select a sandbox provider

`CompanionSandboxAdapter` binds a host-supplied sandbox service to the stable provider protocol. `DaytonaSandboxProvider` is available through `anycode-py[sandbox]`; it supports lifecycle, commands, files, streaming, cancellation, evidence digests, and cleanup through the maintained async SDK. Its capability report marks stable snapshots unsupported, so callers cannot accidentally depend on a provider feature the adapter cannot prove.

`PolicySandboxProvider` applies external policy before sandbox actions. A provider capability report is descriptive, not an isolation certification. The provider and host remain responsible for kernel or VM isolation, egress enforcement, image provenance, tenant separation, storage erasure, and secret injection.

See [Run work through sandbox providers](sandbox-providers.md) for Daytona setup, commands, streaming, file transfer, secrets, cancellation, and companion-service integration.

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

## The complete, runnable program

The sections above each cover one control. Here is a single file that ties them together and runs offline with no provider key. It builds one immutable `ExecutionContext`, enforces external policy at the model boundary, routes to a region- and classification-compatible provider, maps a runtime event to telemetry without exporting prompt content, carries the same identity across the sandbox boundary, and drains a host before publishing an A2A Agent Card. One identity envelope flows through every boundary.

```python title="portable_infrastructure.py"
import asyncio

from anycode import (
    A2A_AGENT_CARD_PATH,
    CapabilityDescriptor,
    ExecutionContext,
    GenAITelemetryConfig,
    GenAITelemetryMapper,
    HostLifecycle,
    ModelRoutingRequest,
    PolicyDecision,
    PolicyEnforcer,
    PolicyRequest,
    PolicyRouter,
    ProviderCapabilityDescriptor,
    SandboxSpec,
    build_deployment_agent_card,
    uuid7,
)


class RegionPolicy:
    """External policy adapter returning the versioned decision contract."""

    async def decide(self, request: PolicyRequest) -> PolicyDecision:
        allowed = request.context.required_region in request.context.allowed_regions
        return PolicyDecision(
            id=str(uuid7()),
            run_id=request.run_id,
            task_id=request.task_id,
            outcome="allow" if allowed else "deny",
            policy_version="region-policy/1",
            reason_codes=("region_allowed" if allowed else "region_denied",),
            correlation_id=request.correlation_id,
            causation_id=request.causation_id,
            generation=request.generation,
            attempt=request.attempt,
        )


def build_context() -> ExecutionContext:
    return ExecutionContext(
        principal="user:reviewer-42",
        subject="document:release-notes",
        workload_identity="kubernetes:serviceaccount:anycode-service",
        tenant_scope="tenant:example",
        classification="confidential",
        allowed_regions=("eu-west",),
        required_region="eu-west",
        credential_references=("vault:providers/reviewer",),
    )


async def main() -> None:
    context = build_context()

    # 1. Identity plus external policy at the model boundary, failing closed.
    enforcer = PolicyEnforcer(RegionPolicy(), fail_closed=True)
    decision = await enforcer.enforce(
        PolicyRequest(
            run_id="run-42",
            action="model.invoke",
            resource="model:secure-eu",
            boundary="model",
            context=context,
            correlation_id="correlation-42",
        )
    )
    if not decision.allowed:
        raise PermissionError(decision.error.message if decision.error else "denied")
    print("policy:", decision.decision.outcome, decision.decision.reason_codes)

    # 2. Route to a provider that satisfies region, classification, and capability.
    router = PolicyRouter(
        (
            ProviderCapabilityDescriptor(
                provider="provider-eu",
                model="secure-eu",
                context_window=128_000,
                structured_output=True,
                tool_use=True,
                regions=("eu-west",),
                allowed_classifications=("public", "internal", "confidential"),
                compatibility_class="chat-json-v1",
            ),
        )
    )
    route = router.route(
        ModelRoutingRequest(
            task_id="task-review",
            structured_output=True,
            classification=context.classification,
            required_region=context.required_region,
        )
    )
    print("route:", route.selected_provider, route.selected_model)

    # 3. Map a runtime event to telemetry without exporting prompt content.
    telemetry = GenAITelemetryMapper(GenAITelemetryConfig(profile="metadata")).map(
        "model.completed",
        {
            "provider": route.selected_provider,
            "model": route.selected_model,
            "prompt": "not exported",
            "input_tokens": 10,
        },
        context=context,
    )
    print("telemetry mapped:", telemetry is not None)

    # 4. The same immutable identity envelope crosses the sandbox boundary.
    sandbox_request = SandboxSpec(
        run_id="run-42",
        task_id="task-review",
        correlation_id="correlation-42",
        context=context,
        image="python:3.12-slim",
        network="allowlist",
        allowed_domains=("pypi.org",),
    )
    print("sandbox tenant:", sandbox_request.context.tenant_scope)

    # 5. Host admission, graceful drain, and A2A discovery.
    lifecycle = HostLifecycle(max_inflight=50)
    await lifecycle.start()
    admission = await lifecycle.admit("work-42")
    if admission.accepted and await lifecycle.begin("work-42"):
        await lifecycle.complete("work-42")
    drain = await lifecycle.drain(timeout_seconds=5)
    print("host drained cleanly:", drain.drained)

    card = build_deployment_agent_card(
        CapabilityDescriptor(
            name="release-review-service",
            implementation_version="0.8.0",
            operations=("review.submit", "review.status"),
        ),
        endpoint="https://agents.example.com/release-review",
        description="Reviews release changes against repository policy.",
        organization="Example Engineering",
    )
    print("agent card:", A2A_AGENT_CARD_PATH, "->", card.supported_interfaces[0].url)


if __name__ == "__main__":
    asyncio.run(main())
```

Run it from the project root:

```bash
uv run python portable_infrastructure.py
```

!!! tip "Tested copy"
    See [`examples/40_operational_portability.py`](https://github.com/Quantlix/anycode/blob/main/examples/40_operational_portability.py), the CI-tested program this walkthrough is based on. It runs the identity, policy, routing, and telemetry core without credentials.

For concrete configuration, continue with [durability backends](durability-backends.md), [policy routing](policy-routing.md), [service hosting](hosting-services.md), and [GenAI telemetry](genai-telemetry.md).
