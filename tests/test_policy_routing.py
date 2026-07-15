"""Policy-driven routing restriction and fallback tests."""

from anycode.routing import ModelRoutingRequest, PolicyRouter, ProviderCapabilityDescriptor


def _router() -> PolicyRouter:
    return PolicyRouter(
        (
            ProviderCapabilityDescriptor(
                provider="provider-a",
                model="text-us",
                context_window=100_000,
                structured_output=True,
                tool_use=True,
                regions=("us",),
                allowed_classifications=("public", "internal"),
                compatibility_class="chat-json-v1",
                input_cost_per_million=1.0,
                output_cost_per_million=2.0,
                typical_latency_ms=100,
            ),
            ProviderCapabilityDescriptor(
                provider="provider-b",
                model="secure-eu",
                modalities=("text", "image"),
                context_window=200_000,
                structured_output=True,
                tool_use=True,
                regions=("eu",),
                allowed_classifications=("public", "internal", "confidential"),
                compatibility_class="chat-json-v1",
                input_cost_per_million=2.0,
                output_cost_per_million=3.0,
                typical_latency_ms=200,
            ),
            ProviderCapabilityDescriptor(
                provider="provider-c",
                model="incompatible-eu",
                context_window=200_000,
                structured_output=True,
                tool_use=True,
                regions=("eu",),
                allowed_classifications=("public", "internal", "confidential"),
                compatibility_class="responses-v2",
                input_cost_per_million=0.1,
                output_cost_per_million=0.1,
                typical_latency_ms=50,
            ),
        )
    )


def test_routing_filters_region_classification_capability_and_fallback_class() -> None:
    decision = _router().route(
        ModelRoutingRequest(
            task_id="task-1",
            modalities=("text", "image"),
            context_tokens=50_000,
            expected_output_tokens=2_000,
            structured_output=True,
            tool_use=True,
            classification="confidential",
            required_region="eu",
            fallback_compatibility_class="chat-json-v1",
        )
    )

    assert decision.selected_provider == "provider-b" and decision.selected_model == "secure-eu"
    assert decision.fallback and decision.compatibility_class == "chat-json-v1"
    assessments = {candidate.provider: candidate for candidate in decision.assessments}
    assert "classification_restricted" in assessments["provider-a"].rejection_reasons
    assert "region_unavailable" in assessments["provider-a"].rejection_reasons
    assert "fallback_incompatible" in assessments["provider-c"].rejection_reasons


def test_routing_never_crosses_provider_budget_latency_or_health_restrictions() -> None:
    decision = _router().route(
        ModelRoutingRequest(
            task_id="task-2",
            classification="confidential",
            required_region="eu",
            allowed_providers=("provider-b",),
            max_cost_usd=0.000001,
            max_latency_ms=100,
        )
    )

    assert decision.selected_model is None and decision.error is not None and decision.error.code == "no_eligible_model"
    assessments = {candidate.provider: candidate for candidate in decision.assessments}
    assert "provider_not_allowed" in assessments["provider-a"].rejection_reasons
    assert "cost_limit_exceeded" in assessments["provider-b"].rejection_reasons
    assert "latency_limit_exceeded" in assessments["provider-b"].rejection_reasons


def test_routing_decision_records_cost_and_is_deterministic() -> None:
    request = ModelRoutingRequest(task_id="task-3", context_tokens=1_000, expected_output_tokens=1_000)
    first = _router().route(request)
    second = _router().route(request)

    assert first == second
    assert first.selected_provider == "provider-c"
    selected = next(candidate for candidate in first.assessments if candidate.provider == "provider-c")
    assert selected.estimated_cost_usd == 0.0002
