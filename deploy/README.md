# Reference deployment profiles

`container/compose.yaml` is a single-replica generic-container profile with a persistent volume. `kubernetes/` is a multi-replica profile using Dapr state, a service account for workload identity, resource limits, probes, drain hooks, and OTLP configuration. Replace the example image with a conformant image built under `IMAGE_CONTRACT.md`.

Before an upgrade, verify the new runtime reads the persisted contract and backend versions, run the service/A2A conformance suite against a canary endpoint, and preserve the old image digest. Use a rolling strategy with `maxUnavailable: 0`; readiness removes draining pods before termination, while the external backend preserves admitted work and resumable event cursors. Do not run old and new versions together if their supported contract-version intersection is empty.

Rollback by restoring the previous immutable image digest without reverting durable state. A release that writes a new schema must provide a backward-readable window or an explicit export/migration step before rollout. Never roll back by deleting events, checkpoints, artifacts, fencing generations, or migration records. Agent Cards are generated from each deployment’s public endpoint, so canary and production cards intentionally have different interface URLs.

Host guarantees include process placement, TLS, ingress identity, service account issuance, secret delivery, network policy, volumes, and termination timing. AnyCode guarantees only the semantic behavior exposed by its runtime, configured backend, policy, artifact, telemetry, sandbox, and protocol capability reports.

