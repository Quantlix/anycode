"""Human-in-the-loop approval gates for agent workflows."""

from anycode.hitl.approval import ApprovalManager
from anycode.hitl.channels import CallbackApprovalGate, StdinApprovalGate, WebhookApprovalGate

__all__ = [
    "ApprovalManager",
    "CallbackApprovalGate",
    "StdinApprovalGate",
    "WebhookApprovalGate",
]
