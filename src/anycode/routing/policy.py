"""Policy-driven provider capability filtering and inspectable routing."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from anycode.contracts.models import ArtifactClassification, ContractError, ContractModel

Modality = Literal["text", "image", "audio", "video", "embedding"]


class ProviderCapabilityDescriptor(ContractModel):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    modalities: tuple[Modality, ...] = ("text",)
    context_window: int = Field(ge=1)
    structured_output: bool = False
    tool_use: bool = False
    regions: tuple[str, ...] = ()
    allowed_classifications: tuple[ArtifactClassification, ...] = ("public", "internal")
    compatibility_class: str = Field(min_length=1)
    input_cost_per_million: float = Field(default=0.0, ge=0)
    output_cost_per_million: float = Field(default=0.0, ge=0)
    typical_latency_ms: float = Field(default=0.0, ge=0)
    healthy: bool = True


class ModelRoutingRequest(ContractModel):
    task_id: str = Field(min_length=1)
    modalities: tuple[Modality, ...] = ("text",)
    context_tokens: int = Field(default=1, ge=1)
    expected_output_tokens: int = Field(default=1, ge=1)
    structured_output: bool = False
    tool_use: bool = False
    classification: ArtifactClassification = "internal"
    required_region: str | None = None
    allowed_providers: tuple[str, ...] = ()
    denied_providers: tuple[str, ...] = ()
    max_cost_usd: float | None = Field(default=None, ge=0)
    budget_remaining_usd: float | None = Field(default=None, ge=0)
    max_latency_ms: float | None = Field(default=None, ge=0)
    fallback_compatibility_class: str | None = None


class CandidateAssessment(ContractModel):
    provider: str
    model: str
    eligible: bool
    rejection_reasons: tuple[str, ...] = ()
    estimated_cost_usd: float = Field(ge=0)
    typical_latency_ms: float = Field(ge=0)
    compatibility_class: str


class ModelRoutingDecision(ContractModel):
    task_id: str
    selected_provider: str | None = None
    selected_model: str | None = None
    compatibility_class: str | None = None
    fallback: bool = False
    assessments: tuple[CandidateAssessment, ...]
    error: ContractError | None = None


class PolicyRouter:
    """Filters hard policy constraints before deterministic cost/latency selection."""

    def __init__(self, descriptors: tuple[ProviderCapabilityDescriptor, ...]) -> None:
        if not descriptors:
            raise ValueError("At least one provider capability descriptor is required")
        self._descriptors = descriptors

    @staticmethod
    def _estimated_cost(descriptor: ProviderCapabilityDescriptor, request: ModelRoutingRequest) -> float:
        return (
            descriptor.input_cost_per_million * request.context_tokens + descriptor.output_cost_per_million * request.expected_output_tokens
        ) / 1_000_000

    def route(self, request: ModelRoutingRequest) -> ModelRoutingDecision:
        assessments: list[CandidateAssessment] = []
        for descriptor in self._descriptors:
            reasons: list[str] = []
            cost = self._estimated_cost(descriptor, request)
            if not descriptor.healthy:
                reasons.append("unhealthy")
            if request.allowed_providers and descriptor.provider not in request.allowed_providers:
                reasons.append("provider_not_allowed")
            if descriptor.provider in request.denied_providers:
                reasons.append("provider_denied")
            if request.classification not in descriptor.allowed_classifications:
                reasons.append("classification_restricted")
            if request.required_region and request.required_region not in descriptor.regions:
                reasons.append("region_unavailable")
            if not set(request.modalities).issubset(descriptor.modalities):
                reasons.append("modality_unsupported")
            if request.context_tokens > descriptor.context_window:
                reasons.append("context_window_exceeded")
            if request.structured_output and not descriptor.structured_output:
                reasons.append("structured_output_unsupported")
            if request.tool_use and not descriptor.tool_use:
                reasons.append("tool_use_unsupported")
            if request.max_cost_usd is not None and cost > request.max_cost_usd:
                reasons.append("cost_limit_exceeded")
            if request.budget_remaining_usd is not None and cost > request.budget_remaining_usd:
                reasons.append("budget_exceeded")
            if request.max_latency_ms is not None and descriptor.typical_latency_ms > request.max_latency_ms:
                reasons.append("latency_limit_exceeded")
            if request.fallback_compatibility_class and descriptor.compatibility_class != request.fallback_compatibility_class:
                reasons.append("fallback_incompatible")
            assessments.append(
                CandidateAssessment(
                    provider=descriptor.provider,
                    model=descriptor.model,
                    eligible=not reasons,
                    rejection_reasons=tuple(reasons),
                    estimated_cost_usd=cost,
                    typical_latency_ms=descriptor.typical_latency_ms,
                    compatibility_class=descriptor.compatibility_class,
                )
            )
        eligible = [assessment for assessment in assessments if assessment.eligible]
        if not eligible:
            return ModelRoutingDecision(
                task_id=request.task_id,
                fallback=request.fallback_compatibility_class is not None,
                assessments=tuple(assessments),
                error=ContractError(code="no_eligible_model", message="No model satisfies all routing constraints."),
            )
        selected = min(
            eligible,
            key=lambda candidate: (candidate.estimated_cost_usd, candidate.typical_latency_ms, candidate.provider, candidate.model),
        )
        return ModelRoutingDecision(
            task_id=request.task_id,
            selected_provider=selected.provider,
            selected_model=selected.model,
            compatibility_class=selected.compatibility_class,
            fallback=request.fallback_compatibility_class is not None,
            assessments=tuple(assessments),
        )
