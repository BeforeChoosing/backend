from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AuditEventRequest(BaseModel):
    action: str = Field(min_length=1, max_length=120)
    target: str = Field(default="", max_length=160)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuditEventResponse(BaseModel):
    recorded: bool
    event_id: str | None = None


class AuditUsageSummary(BaseModel):
    app_mode: str
    event_count: int
    mean_duration_ms: float
    model_mean_duration_ms: float = 0.0
    model_call_count: int
    input_tokens: int
    output_tokens: int
    paths: list[dict[str, Any]] = Field(default_factory=list)
