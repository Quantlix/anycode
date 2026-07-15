# AnyCode service image contract

An AnyCode service image is conformant when it packages Python 3.12+, the service/A2A profile, and the selected durability adapter, and runs as a non-root user with an immutable root filesystem. The image entry point is `anycode-service`; it must bind `0.0.0.0:${PORT:-8080}` and expose:

- `GET /health/live` for process liveness;
- `GET /health/ready` for admission readiness;
- `POST /health/drain` to stop admission and drain or durably return accepted work;
- `GET /.well-known/agent-card.json` with an A2A 1.0 card generated for the externally visible endpoint;
- the versioned AnyCode service and A2A operations advertised by its capability descriptor.

The process must handle `SIGTERM` as a drain request and exit before `ANYCODE_DRAIN_TIMEOUT_SECONDS` plus the platform termination margin. It must not report readiness while starting, draining, unable to reach a required policy service, or unable to reach its durable backend. Liveness must not depend on telemetry availability.

Configuration is supplied through environment variables or mounted files. Required categories are service endpoint, backend and state store, workload-identity reference, secret references, artifact store, request/stream/artifact limits, OTLP endpoint and capture profile, and drain timeout. Variables ending in `_REF` contain provider references, never secret values. The image must not bake API keys, tenant credentials, writable run state, or deployment-specific Agent Cards into a layer.

Local SQLite is a single-replica profile. Multi-replica profiles use a backend whose capability report includes external persistence, fencing, durable wakes, and external signals. The host supplies TLS, ingress authentication, network policy, workload identity, key management, and secret resolution; AnyCode supplies semantic run state, stream cursors, cancellation, and backend fencing.

At release time, build the image from the wheel produced by the same source revision, pin the base image by digest, emit an SBOM and provenance, scan both base and Python dependencies, and publish the image and wheel versions together. Deployment manifests must use an immutable digest or version tag, never `latest`.

