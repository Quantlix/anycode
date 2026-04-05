"""Deterministic round-trip serialization for CheckpointData and LLMMessage content blocks."""

from __future__ import annotations

import json
from typing import Any

from anycode.constants import (
    BLOCK_TYPE_IMAGE,
    BLOCK_TYPE_TEXT,
    BLOCK_TYPE_TOOL_RESULT,
    BLOCK_TYPE_TOOL_USE,
    CHECKPOINT_FORMAT_VERSION,
)
from anycode.types import (
    AgentRunResult,
    CheckpointData,
    ContentBlock,
    ImageBlock,
    ImageSource,
    LLMMessage,
    Task,
    TextBlock,
    TokenUsage,
    ToolCallRecord,
    ToolResultBlock,
    ToolUseBlock,
)


def serialize_checkpoint(data: CheckpointData) -> str:
    payload = {
        "id": data.id,
        "workflow_id": data.workflow_id,
        "version": data.version,
        "wave_index": data.wave_index,
        "total_token_usage": data.total_token_usage.model_dump(mode="json"),
        "created_at": data.created_at.isoformat(),
        "metadata": data.metadata,
        "tasks": [_serialize_task(t) for t in data.tasks],
        "agent_results": {k: _serialize_agent_result(v) for k, v in data.agent_results.items()},
    }
    return json.dumps(payload, indent=2, default=str)


def deserialize_checkpoint(raw: str) -> CheckpointData:
    data = json.loads(raw)
    return CheckpointData(
        id=data["id"],
        workflow_id=data["workflow_id"],
        version=data.get("version", CHECKPOINT_FORMAT_VERSION),
        wave_index=data["wave_index"],
        total_token_usage=TokenUsage(**data["total_token_usage"]),
        created_at=data["created_at"],
        metadata=data.get("metadata"),
        tasks=[_deserialize_task(t) for t in data["tasks"]],
        agent_results={k: _deserialize_agent_result(v) for k, v in data["agent_results"].items()},
    )


def _serialize_task(task: Task) -> dict[str, Any]:
    return task.model_dump(mode="json")


def _deserialize_task(data: dict[str, Any]) -> Task:
    return Task(**data)


def _serialize_agent_result(result: AgentRunResult) -> dict[str, Any]:
    return {
        "success": result.success,
        "output": result.output,
        "token_usage": result.token_usage.model_dump(mode="json"),
        "tool_calls": [tc.model_dump(mode="json") for tc in result.tool_calls],
        "messages": [_serialize_message(m) for m in result.messages],
    }


def _deserialize_agent_result(data: dict[str, Any]) -> AgentRunResult:
    return AgentRunResult(
        success=data["success"],
        output=data["output"],
        token_usage=TokenUsage(**data["token_usage"]),
        tool_calls=[ToolCallRecord(**tc) for tc in data.get("tool_calls", [])],
        messages=[_deserialize_message(m) for m in data.get("messages", [])],
    )


def _serialize_message(msg: LLMMessage) -> dict[str, Any]:
    return {
        "role": msg.role,
        "content": [_serialize_content_block(b) for b in msg.content],
    }


def _deserialize_message(data: dict[str, Any]) -> LLMMessage:
    return LLMMessage(
        role=data["role"],
        content=[_deserialize_content_block(b) for b in data["content"]],
    )


def _serialize_content_block(block: ContentBlock) -> dict[str, Any]:
    return block.model_dump(mode="json")


def _deserialize_content_block(data: dict[str, Any]) -> ContentBlock:
    block_type = data.get("type")
    if block_type == BLOCK_TYPE_TEXT:
        return TextBlock(**data)
    if block_type == BLOCK_TYPE_TOOL_USE:
        return ToolUseBlock(**data)
    if block_type == BLOCK_TYPE_TOOL_RESULT:
        return ToolResultBlock(**data)
    if block_type == BLOCK_TYPE_IMAGE:
        return ImageBlock(type=BLOCK_TYPE_IMAGE, source=ImageSource(**data["source"]))
    raise ValueError(f"Unknown content block type: {block_type!r}")
