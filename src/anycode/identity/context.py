"""Validated execution identity propagated across runtime boundaries."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

_SENSITIVE_KEY_PARTS = ("api_key", "authorization", "credential", "password", "private_key", "secret", "token")
_SECRET_VALUE_PATTERN = re.compile(
    r"(?:\bBearer\s+[A-Za-z0-9._~+/=-]{8,}|\bsk-[A-Za-z0-9_-]{12,}|\bgh[pousr]_[A-Za-z0-9_]{12,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)",
    re.IGNORECASE,
)


ArtifactClassification = Literal["public", "internal", "confidential", "restricted"]


class IdentityModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DelegationGrant(IdentityModel):
    delegator: str = Field(min_length=1)
    delegatee: str = Field(min_length=1)
    scopes: tuple[str, ...] = ()
    expires_at: datetime | None = None


class ExecutionContext(IdentityModel):
    """Portable identity, tenancy, classification, and placement constraints."""

    principal: str = Field(min_length=1)
    subject: str | None = None
    workload_identity: str | None = None
    tenant_scope: str = Field(default="default", min_length=1)
    delegation: tuple[DelegationGrant, ...] = ()
    classification: ArtifactClassification = "internal"
    allowed_regions: tuple[str, ...] = ()
    required_region: str | None = None
    credential_references: tuple[str, ...] = ()
    trace_id: str | None = None
    attributes: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_context(self) -> ExecutionContext:
        if self.required_region is not None and self.allowed_regions and self.required_region not in self.allowed_regions:
            raise ValueError("required_region must be present in allowed_regions")
        if len(set(self.allowed_regions)) != len(self.allowed_regions):
            raise ValueError("allowed_regions must not contain duplicates")
        if any(not reference or ":" not in reference for reference in self.credential_references):
            raise ValueError("credential references must use a provider-qualified form such as 'env:OPENAI_API_KEY'")
        self._reject_raw_credentials(self.attributes)
        return self

    @classmethod
    def _reject_raw_credentials(cls, value: JsonValue, *, path: str = "attributes") -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                normalized = key.casefold().replace("-", "_")
                if any(part in normalized for part in _SENSITIVE_KEY_PARTS):
                    raise ValueError(f"Raw credential-like field is not allowed in execution context: {path}.{key}")
                cls._reject_raw_credentials(nested, path=f"{path}.{key}")
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                cls._reject_raw_credentials(nested, path=f"{path}[{index}]")
        elif isinstance(value, str) and _SECRET_VALUE_PATTERN.search(value):
            raise ValueError(f"Raw credential-like value is not allowed in execution context: {path}")

    def audit_attributes(self) -> dict[str, str]:
        """Return low-cardinality identity metadata with no credential references."""
        attributes = {
            "principal": self.principal,
            "tenant_scope": self.tenant_scope,
            "classification": self.classification,
        }
        if self.subject:
            attributes["subject"] = self.subject
        if self.workload_identity:
            attributes["workload_identity"] = self.workload_identity
        if self.required_region:
            attributes["region"] = self.required_region
        return attributes

    def policy_json(self) -> str:
        """Stable JSON representation suitable for external policy input."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


BoundaryKind = Literal["model", "tool", "sandbox", "a2a", "artifact", "backend"]
