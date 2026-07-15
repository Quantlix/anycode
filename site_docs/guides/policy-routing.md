---
title: "Route Models With AnyCode Policy Constraints"
description: Filter AI models by data classification, region, modality, capabilities, budget, latency, and compatibility before deterministic selection.
keywords: AnyCode model routing, policy based LLM routing, AI model cost routing, regional LLM routing, model fallback policy
---

# Route models with hard policy constraints

`PolicyRouter` rejects models that violate required policy or runtime constraints, then selects the cheapest eligible candidate. Latency, provider name, and model name provide deterministic tie-breakers. The decision includes an assessment for every candidate, so a caller can explain both the selection and every rejection.

## Describe available providers

Create one `ProviderCapabilityDescriptor` for each model available to the deployment. Keep these descriptors in deployment configuration and update health and cost data from controlled operational sources.

```python
from anycode import ProviderCapabilityDescriptor

models = (
    ProviderCapabilityDescriptor(
        provider="regional-provider",
        model="review-small",
        modalities=("text",),
        context_window=64_000,
        structured_output=True,
        tool_use=True,
        regions=("eu-west",),
        allowed_classifications=("public", "internal", "confidential"),
        compatibility_class="review-v1",
        input_cost_per_million=0.25,
        output_cost_per_million=1.00,
        typical_latency_ms=450,
    ),
    ProviderCapabilityDescriptor(
        provider="general-provider",
        model="review-large",
        modalities=("text", "image"),
        context_window=128_000,
        structured_output=True,
        tool_use=True,
        regions=("us-east", "eu-west"),
        allowed_classifications=("public", "internal"),
        compatibility_class="review-v1",
        input_cost_per_million=2.00,
        output_cost_per_million=8.00,
        typical_latency_ms=900,
    ),
)
```

`healthy=False` removes a model from eligibility without deleting its assessment. A descriptor is deployment evidence, not provider discovery; the host remains responsible for refreshing it when pricing, placement, or health changes.

## Route one request

Put non-negotiable requirements in `ModelRoutingRequest`. The router never relaxes them to force a result.

```python
from anycode import ModelRoutingRequest, PolicyRouter

router = PolicyRouter(models)
decision = router.route(
    ModelRoutingRequest(
        task_id="task-review",
        modalities=("text",),
        context_tokens=20_000,
        expected_output_tokens=2_000,
        structured_output=True,
        tool_use=True,
        classification="confidential",
        required_region="eu-west",
        max_cost_usd=0.01,
        budget_remaining_usd=0.50,
        max_latency_ms=1_000,
    )
)

if decision.error:
    reasons = {
        f"{item.provider}/{item.model}": item.rejection_reasons
        for item in decision.assessments
    }
    raise RuntimeError(f"{decision.error.code}: {reasons}")

print(decision.selected_provider, decision.selected_model)
```

Cost is estimated from requested input and output tokens and the descriptor's per-million-token prices. Treat it as an admission estimate, then use actual provider usage for final accounting.

## Understand hard filters

| Request field | Candidate requirement | Rejection reason |
| --- | --- | --- |
| `allowed_providers`, `denied_providers` | Provider passes both lists | `provider_not_allowed` or `provider_denied` |
| `classification` | Classification appears in the model's allowed set | `classification_restricted` |
| `required_region` | Region appears in the model's regions | `region_unavailable` |
| `modalities` | Every requested modality is supported | `modality_unsupported` |
| `context_tokens` | Request fits the context window | `context_window_exceeded` |
| `structured_output`, `tool_use` | Requested capabilities are enabled | `structured_output_unsupported` or `tool_use_unsupported` |
| `max_cost_usd`, `budget_remaining_usd` | Estimated cost stays within both limits | `cost_limit_exceeded` or `budget_exceeded` |
| `max_latency_ms` | Typical latency stays within the limit | `latency_limit_exceeded` |
| `fallback_compatibility_class` | Compatibility class matches exactly | `fallback_incompatible` |

When no candidate passes every filter, the router returns `no_eligible_model`. It does not silently use a denied provider, cross a region boundary, downgrade required capabilities, or exceed a budget.

## Keep fallback behavior compatible

Set `fallback_compatibility_class` when retry logic needs another model with the same input, tool, and output contract. This field is an additional hard constraint, not permission to relax the original request.

```python
fallback = router.route(
    ModelRoutingRequest(
        task_id="task-review-retry",
        structured_output=True,
        classification="internal",
        fallback_compatibility_class="review-v1",
        denied_providers=(decision.selected_provider,) if decision.selected_provider else (),
    )
)
```

The returned decision sets `fallback=True` whenever a fallback compatibility class was requested. The caller still owns retry limits, idempotency, and deciding whether a failed operation is safe to repeat.

## Separate task routing from model policy

`DefaultRouter` assigns work to agents by scheduling strategy. `PolicyRouter` selects an eligible provider and model for a specific model request. Use both when a team first chooses an agent by capability and that agent then needs deployment-policy-aware model selection.

## Next steps

- [Propagate execution identity and policy](execution-identity.md)
- [Track and cap model cost](cost-tracking.md)
- [Configure model providers](providers.md)
- [Review basic task routing](routing.md)
