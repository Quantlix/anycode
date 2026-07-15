---
title: "Host AnyCode Services Safely"
description: Add readiness, admission limits, graceful drain, durable return, A2A Agent Cards, and container or Kubernetes deployment contracts to AnyCode services.
keywords: host AnyCode service, Kubernetes AI agents, graceful agent shutdown, A2A Agent Card, containerized agent service
---

# Host AnyCode services safely

`HostLifecycle` is a web-framework-neutral admission and drain controller. It stops new work before shutdown, tracks admitted and running work, and can return unfinished identifiers to durable storage when a termination window expires. The hosting helpers also generate one A2A 1.0 Agent Card for each externally reachable deployment endpoint.

## Add lifecycle control to a service

```python
from anycode import HostLifecycle

lifecycle = HostLifecycle(max_inflight=100)
await lifecycle.start()

admission = await lifecycle.admit("work-42")
if not admission.accepted:
    raise RuntimeError(admission.error.message if admission.error else "not admitted")

if await lifecycle.begin("work-42"):
    try:
        await process_work("work-42")
    finally:
        await lifecycle.complete("work-42")
```

A repeated admission for the same identifier returns `accepted=True` and `duplicate=True`. New work receives a retryable `host_capacity` error at the configured limit or a retryable `host_draining` error unless the lifecycle is ready.

## Map lifecycle state to health endpoints

Expose the image contract through the web framework selected by the application:

| Endpoint | Lifecycle behavior |
| --- | --- |
| `GET /health/live` | Success while `lifecycle.live()` is true |
| `GET /health/ready` | Success only while `lifecycle.ready()` is true and required dependencies are ready |
| `POST /health/drain` | Call `lifecycle.drain(...)`, stop admission, and report the drain result |
| `GET /.well-known/agent-card.json` | Return the deployment-specific Agent Card |

Readiness should also fail when a required policy service or durability backend is unavailable. Liveness must not depend on telemetry export.

## Drain or durably return accepted work

```python
from anycode import WorkItem

accepted_work: dict[str, WorkItem] = load_admitted_work()


async def return_to_backend(work_ids: tuple[str, ...]) -> None:
    for work_id in work_ids:
        await backend.enqueue(accepted_work[work_id])


result = await lifecycle.drain(
    timeout_seconds=25,
    durable_return=return_to_backend,
)

if not result.drained:
    raise RuntimeError(f"work still admitted: {result.remaining}")
```

Drain changes the host state to `draining` immediately. If admitted work completes within the timeout, the host stops cleanly. On timeout, a successful `durable_return` callback removes the returned identifiers and stops the host. Without that callback, the result reports `drained=False` and preserves the remaining identifiers for the caller to handle.

The callback must be idempotent. A process can lose its final acknowledgement after durable state has changed.

## Generate a deployment Agent Card

```python
from anycode import (
    A2A_AGENT_CARD_PATH,
    CapabilityDescriptor,
    build_deployment_agent_card,
)

capability = CapabilityDescriptor(
    name="release-review-service",
    implementation_version="0.8.0",
    operations=("review.submit", "review.status", "review.cancel"),
    supports_cancellation=True,
    supports_resume=True,
    supports_event_stream=True,
)

card = build_deployment_agent_card(
    capability,
    endpoint="https://agents.example.com/release-review",
    description="Reviews release changes against repository policy.",
    organization="Example Engineering",
    organization_url="https://example.com",
    openid_connect_url="https://identity.example.com/.well-known/openid-configuration",
)

assert A2A_AGENT_CARD_PATH == "/.well-known/agent-card.json"
payload = card.model_dump(mode="json", by_alias=True)
```

The card advertises the service interface at `<endpoint>/a2a` and creates one skill per capability operation. Generate it at runtime because production and canary deployments have different public endpoints. Do not bake a deployment URL or security configuration into the image.

## Choose a deployment profile

| Profile | Replicas | Durability | Use when |
| --- | --- | --- | --- |
| Generic container | One | SQLite on a persistent volume | A bounded service runs on one host and local persistence is acceptable |
| Kubernetes with Dapr | Multiple | External Dapr state store | Rolling replicas need shared claims, fencing, wakes, and signals |

Reference manifests live under `deploy/container/` and `deploy/kubernetes/`. Replace example images with a version tag or digest built from the same revision as the released wheel. Never deploy `latest`.

## Meet the image contract

A conformant image uses Python 3.12 or newer, runs as a non-root user, prefers an immutable root filesystem, binds `0.0.0.0:${PORT:-8080}`, and handles `SIGTERM` as a drain request. The host supplies TLS, ingress authentication, workload identity, secret delivery, network policy, storage, and termination timing.

Before a rolling upgrade:

1. Verify the new runtime can read deployed contract and backend versions.
2. Run service and A2A conformance against a canary endpoint.
3. Preserve the previous immutable image digest.
4. Stop admission before termination and durably return work that misses the window.
5. Roll back the image without deleting or reversing durable history.

See `deploy/IMAGE_CONTRACT.md` and `deploy/README.md` for the complete host and rollback requirements.

## Next steps

- [Configure durability backends](durability-backends.md)
- [Map GenAI telemetry safely](genai-telemetry.md)
- [Review production readiness](production-readiness.md)
- [Deploy portable infrastructure](portable-infrastructure.md)
