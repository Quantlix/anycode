"""Inspect identity-aware policy, telemetry capture, and model routing."""

from __future__ import annotations

import asyncio
import json

from anycode.identity.context import ExecutionContext
from anycode.identity.policy import PolicyEnforcer, PolicyRequest
from anycode.routing import ModelRoutingRequest, PolicyRouter, ProviderCapabilityDescriptor
from anycode.telemetry import GenAITelemetryConfig, GenAITelemetryMapper


async def main() -> None:
    context = ExecutionContext(
        principal="user:example",
        workload_identity="spiffe://example/agent",
        tenant_scope="tenant-demo",
        classification="confidential",
        allowed_regions=("eu",),
        required_region="eu",
        credential_references=("env:OPENAI_API_KEY",),
    )
    policy = PolicyEnforcer(fail_closed=False)
    decision = await policy.enforce(
        PolicyRequest(
            run_id="run-demo",
            action="invoke",
            resource="model:secure-eu",
            boundary="model",
            context=context,
            correlation_id="corr-demo",
        )
    )
    router = PolicyRouter(
        (
            ProviderCapabilityDescriptor(
                provider="provider-demo",
                model="secure-eu",
                context_window=128_000,
                structured_output=True,
                tool_use=True,
                regions=("eu",),
                allowed_classifications=("public", "internal", "confidential"),
                compatibility_class="chat-json-v1",
            ),
        )
    )
    route = router.route(
        ModelRoutingRequest(
            task_id="task-demo",
            structured_output=True,
            classification=context.classification,
            required_region=context.required_region,
        )
    )
    telemetry = GenAITelemetryMapper(GenAITelemetryConfig(profile="metadata")).map(
        "model.completed",
        {"provider": route.selected_provider, "model": route.selected_model, "prompt": "not exported", "input_tokens": 10},
        context=context,
    )
    print(  # noqa: T201
        json.dumps(
            {
                "policy": decision.model_dump(mode="json"),
                "route": route.model_dump(mode="json"),
                "telemetry": telemetry.model_dump(mode="json") if telemetry else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())

