---
title: "AnyCode Production Readiness Checklist"
description: "A go or no-go checklist for deploying AnyCode with scoped tools, isolated workers, durable state, idempotent side effects, monitoring, and rollback controls."
keywords: AnyCode production readiness, AI agent deployment checklist, agent security checklist, MCP production, LLM operations, go no-go review
---

# Production Readiness Checklist

AnyCode is conditionally suitable for bounded production workloads. A deployment is a **go** only when every mandatory control below is satisfied, no no-go condition applies, and the team has tested rollback and incident response. The package remains alpha: pin the exact version, rely only on the documented compatibility contract, and repeat this review whenever the model, tools, extensions, data, or deployment topology changes.

!!! danger "This checklist is a gate, not a guarantee"
    Passing the checklist does not certify a system or transfer responsibility to the framework. The deploying organization owns infrastructure security, identity, data governance, downstream authorization, legal review, and operational response. Start with the [security and threat model](../reference/security.md).

## Current readiness assessment

| Workload | Current assessment |
| --- | --- |
| Local experiments and deterministic evaluation | Supported. Use fake providers and disposable data where possible. |
| Bounded internal automation with reversible actions | Eligible after every mandatory control passes. Prefer read-only tools and low-sensitivity data for the first deployment. |
| Customer-facing, multi-tenant, or sensitive-data workflows | Conditional. Requires application-owned authentication and tenancy isolation, protected storage, privacy review, stronger abuse testing, and staffed operations. |
| Irreversible external actions | Conditional only with downstream idempotency, durable claim storage, explicit authorization, human approval, and an operator reconciliation runbook. |
| Direct safety-critical control, critical infrastructure control, or autonomous high-impact decisions | No-go as the sole decision or control system. Keep a separate authoritative system and qualified human in control. |

Alpha describes release maturity, not an automatic deployment decision. AnyCode has explicit public API and persisted-format contracts plus production-oriented runtime controls, but pre-1.0 minor releases may contain documented breaking changes. A pinned, tested deployment can be eligible while an unrestricted deployment of the same package remains unsafe.

## Stop conditions

A deployment is a **no-go** when any statement below is true:

- [ ] The process can run unrestricted shell commands, read the host filesystem, or reach arbitrary networks while holding production credentials.
- [ ] A model can trigger an irreversible action without durable idempotency, downstream authorization, and human approval where required.
- [ ] Unreviewed custom Python, plugins, or stdio MCP servers execute inside the trusted worker.
- [ ] Sensitive or regulated data has no approved classification, provider policy, storage protection, retention rule, and deletion process.
- [ ] The workflow directly controls safety-critical equipment, critical infrastructure, or a high-impact decision without a separate authoritative control and qualified human review.
- [ ] Operators cannot stop new runs, revoke credentials, inspect outcomes, reconcile uncertain side effects, and roll back the release.
- [ ] The team cannot reproduce the release from a pinned dependency set or cannot restore its persisted state from backup.

Treat this as a stop list. A checked item blocks release until the architecture changes or an independent authoritative system removes AnyCode from that consequence path.

## Mandatory controls

### 1. Workload and data classification

- [ ] Document every input source, retrieved corpus, model provider, tool, MCP server, plugin, persistence backend, telemetry sink, and external destination.
- [ ] Classify the data that may enter prompts, tool results, memory, checkpoints, logs, traces, and evaluation artifacts.
- [ ] Enumerate every possible external side effect and mark whether it is reversible, idempotent, approval-gated, and auditable.
- [ ] Define the service objective, concurrency target, maximum acceptable spend, recovery time objective, and recovery point objective.
- [ ] Identify tenant boundaries and prompt-injection sources. Decide which content is allowed to influence a tool-enabled agent.
- [ ] Assign a named owner for security, operations, data governance, and each irreversible integration.

### 2. Release and compatibility control

- [ ] Pin the AnyCode version, Python version, selected extras, provider SDKs, MCP SDK, plugins, and transitive dependencies in a reviewed lockfile or immutable image.
- [ ] Build from a trusted source and retain the package or image digest used by the release.
- [ ] Run Ruff, Pyright, the complete non-integration suite, relevant service integrations, and built-wheel smoke tests in the deployment environment.
- [ ] Load representative declarative configs, workflow checkpoints, durable runs, idempotency records, and memory data with the candidate release.
- [ ] Review the [compatibility and versioning policy](../reference/compatibility.md) and release notes for every upgrade.
- [ ] Keep the previous executable and a compatible data snapshot for the full rollback window.

### 3. Worker isolation and identity

- [ ] Run tool-enabled agents as a dedicated non-root identity in a container, VM, or equivalent operating-system boundary.
- [ ] Mount only the required workspace and data. Prefer a read-only root filesystem and an ephemeral writable workspace.
- [ ] Give the worker identity only the API, storage, queue, and network permissions needed by this workflow.
- [ ] Enforce outbound network policy at the firewall, proxy, service mesh, or container layer. Do not rely on prompt instructions or URL validation alone.
- [ ] Inject secrets from an approved secret manager or protected environment. Keep them out of prompts, config files, images, source control, and broad parent environments.
- [ ] Separate development, staging, and production identities, stores, provider keys, MCP endpoints, and telemetry destinations.

### 4. Tool authorization

- [ ] Set `AgentConfig.tools` explicitly for every production agent. Use `tools=[]` for agents that need no ordinary tools.
- [ ] Attach a `ToolSecurityPolicy` to every tool-enabled agent. Set explicit tool allowlists or denylists and filesystem roots.
- [ ] Set `allow_shell=False` unless shell execution is essential and isolated. If enabled, use a narrow executable allowlist and still treat interpreters and launchers as arbitrary-code capability.
- [ ] Set `inherit_environment=False` and allow only the environment variable names required by child processes.
- [ ] Review custom tools as trusted application code. Confirm input validation, authorization, timeouts, cancellation, output limits, and redacted errors.
- [ ] Confirm the downstream service independently authorizes each action. Tool visibility is not a substitute for business authorization.

### 5. MCP and plugin trust

- [ ] Scope each agent to the exact MCP server names it needs. Confirm an agent with omitted or empty `mcp_servers` receives none.
- [ ] Use HTTPS and `allowed_hosts` for remote MCP. Supply bearer credentials through `auth_token_env` and verify missing credentials fail before connection.
- [ ] Enforce DNS and network egress policy outside AnyCode, including private, link-local, metadata-service, and internal control-plane ranges.
- [ ] Set `allow_stdio=False` unless the command and every argument are reviewed operator-controlled code running in an isolated worker.
- [ ] Monitor MCP connection errors and verify the application handles per-server capability loss without unsafe fallback behavior.
- [ ] Pass an explicit fail-closed `PluginTrustPolicy` to production entry-point discovery. Pin and review approved distributions.
- [ ] Treat approved plugins as host-process code with access to files, environment, imports, and networking. Isolate code that cannot receive that trust.

### 6. Side effects and approval

- [ ] Mark every mutating custom tool `side_effecting=True`. Treat all MCP tools as mutating unless a separate operator-owned wrapper proves otherwise.
- [ ] Use SQLite or an application-supplied shared `ToolIdempotencyStore` when a run can restart. Do not use the in-memory store for cross-process or restart guarantees.
- [ ] Carry the same idempotency key into the downstream API whenever that API supports native idempotency.
- [ ] Test replay, key conflict, concurrent claim, process crash, store outage, and completion-record failure.
- [ ] Define a runbook for `side_effect_unknown`. Operators must inspect the external system before completing or deleting an uncertain claim.
- [ ] Require a real human approval handler for sensitive or irreversible actions. Demonstration handlers that auto-approve are not production controls.
- [ ] Put irreversible authorization in the downstream service as well as the agent workflow.

### 7. Data protection and durable state

- [ ] Keep default redaction enabled for telemetry, checkpoints, run stores, memory, evaluation, and evidence unless an independently protected store requires exact replay.
- [ ] Test redaction with organization-specific token formats and sensitive field names. Treat it as a backstop, not data-loss prevention.
- [ ] Confirm that prompts and active in-memory conversations contain only data the selected model provider is approved to receive.
- [ ] Encrypt durable run payloads with an application-supplied `RunPayloadProtector` or an equivalent protected backend. Restrict filesystem and service permissions.
- [ ] Protect workflow checkpoint storage at the storage layer when exact, unredacted replay is enabled.
- [ ] Configure and test retention for runs, transcripts, checkpoints, memory, telemetry, idempotency records, and backups.
- [ ] Test backup, restore, corruption fallback, key rotation, and deletion against representative data volume.
- [ ] For a custom `RunStore`, verify atomic metadata updates, ordered event sequence numbers, checkpoint recovery, and mutually exclusive sweep locks.

### 8. Limits, reliability, and shutdown

- [ ] Set per-agent turn and token limits plus a workflow cost budget with a stop policy appropriate to the workload.
- [ ] Configure provider deadlines, bounded retries, circuit breaking, concurrency limits, request pacing, and queue-wait timeouts.
- [ ] Use a provider gateway or distributed limiter for aggregate multi-process or multi-host quotas. Local capacity controls cover one event loop and scope.
- [ ] Exercise provider timeout, rate limit, malformed response, partial stream, MCP outage, storage outage, and verification failure paths.
- [ ] Verify custom tools and integrations re-raise `asyncio.CancelledError` and release resources in `finally` blocks or async context managers.
- [ ] Call `await engine.close()` or use `async with AnyCode(...)` and verify shutdown drains work, terminates subprocesses, disconnects MCP, and flushes telemetry.
- [ ] Load test at expected concurrency and at the configured overload boundary. Confirm queues remain bounded and rejection is observable.

### 9. Verification and model risk

- [ ] Build deterministic regression scenarios with `FakeAdapter` for routing, tool calls, failures, retries, approvals, cancellation, and terminal outcomes.
- [ ] Test representative prompt injection, malicious retrieved content, forged tool names, path escapes, shell control operators, and oversized output.
- [ ] Use independent computational verification for consequential output. Do not accept model self-evaluation as the final gate.
- [ ] Set verification failures to block or escalate where accepting an unchecked result would be unsafe.
- [ ] Measure quality by workload slice and provider/model version. Define a rollback threshold before release.
- [ ] Keep a qualified human authoritative for high-impact decisions and ambiguous verification outcomes.

### 10. Observability and incident response

- [ ] Correlate run IDs and trace IDs across application logs, JSONL or OTLP telemetry, run stores, provider requests, and downstream side effects.
- [ ] Alert on non-success outcomes, provider unavailability, retry growth, latency, dropped telemetry, missing ingestion, cost thresholds, and every `side_effect_unknown` result.
- [ ] Restrict access to logs, traces, transcripts, and run stores as sensitive data systems.
- [ ] Provide a kill switch that stops admission, cancels active work, disables tools or integrations, and revokes credentials.
- [ ] Write and rehearse runbooks for uncertain side effects, provider compromise, MCP compromise, plugin or dependency compromise, credential leakage, data deletion, and rollback.
- [ ] Define who can approve a restart after an incident and what evidence must be reviewed first.

## Required release evidence

A release decision should link to durable evidence, not a verbal assurance:

| Evidence | Minimum record |
| --- | --- |
| Release identity | AnyCode version, Python version, lockfile or image digest, provider/model identifiers, plugin and MCP inventory. |
| Test gate | CI run plus workload-specific deterministic, integration, abuse, load, cancellation, and recovery results. |
| Data review | Approved classification, provider use, retention, encryption, backup, deletion, and access-control decisions. |
| Side-effect review | Tool inventory, idempotency design, downstream authorization, approval policy, and `side_effect_unknown` drill result. |
| Threat review | Accepted threat-model revision, deployment diagram, trust exceptions, owners, expiry dates, and compensating controls. |
| Operations review | Dashboards, alerts, on-call ownership, kill-switch test, restore drill, credential-revocation drill, and rollback result. |

## Decide go, conditional go, or no-go

**Go** requires all mandatory controls, no stop condition, passing release evidence, a tested rollback, and explicit approval from the workload, security, data, and operations owners.

**Conditional go** still requires every mandatory security, side-effect, data, and recovery control. Use it only for a non-critical operational follow-up with a compensating control, named owner, expiry date, and automatic rollback threshold. It is not a way to waive a stop condition.

**No-go** is the decision when a stop condition applies, a mandatory control is missing, evidence is stale, an uncertain side effect is unresolved, or the team cannot contain and reverse a failed release.

## Reassess after change

Repeat the decision when any of these changes:

- AnyCode, Python, provider SDK, MCP SDK, plugin, or persistence version.
- Model provider, model identifier, system prompt, context source, or verification policy.
- Tool code, side-effect behavior, downstream authorization, MCP server, or plugin allowlist.
- Data classification, tenant model, region, retention, encryption, or telemetry destination.
- Worker identity, network policy, concurrency, process count, host count, or deployment platform.

The useful question is not whether the package is production-ready in the abstract. It is whether this pinned release, workload, data path, extension set, and operating environment meet a reviewed boundary with evidence.

## Next steps

- Read the [security and threat model](../reference/security.md) with the deployment diagram in hand.
- Configure [production controls](production-controls.md) and [tool security](tools.md#enforce-a-workspace-security-policy).
- Review [durable storage](durability.md), [observability](observability.md), [MCP](mcp.md), and [plugins](plugins.md) as separate trust boundaries.
- Use the [compatibility upgrade and rollback procedure](../reference/compatibility.md#upgrade-procedure) for every release.
