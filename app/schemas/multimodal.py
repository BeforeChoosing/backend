from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


MultimodalEvidenceType = Literal["documented_fact", "self_report", "inference"]


class MultimodalEvidenceItem(BaseModel):
    """A candidate citation located on one rendered page."""

    id: str = Field(min_length=1, max_length=160)
    source_ref: str = Field(min_length=1, max_length=220)
    page: int = Field(ge=1)
    bbox: list[int] = Field(min_length=4, max_length=4)
    coordinate_space: Literal["normalized_1000"] = "normalized_1000"
    label: str = Field(min_length=1, max_length=120)
    quote: str = Field(min_length=1, max_length=800)
    evidence_type: MultimodalEvidenceType = "self_report"
    confidence: float = Field(ge=0, le=1)
    status: Literal["candidate", "confirmed", "rejected"] = "candidate"

    @model_validator(mode="after")
    def validate_bbox(self) -> "MultimodalEvidenceItem":
        if any(value < 0 or value > 1000 for value in self.bbox):
            raise ValueError("bbox 必须是 0–1000 的归一化坐标")
        x0, y0, x1, y1 = self.bbox
        if x1 <= x0 or y1 <= y0:
            raise ValueError("bbox 必须满足右下角大于左上角")
        return self


class MultimodalEvidenceResponse(BaseModel):
    file_name: str
    file_sha256: str
    mime_type: str
    page_count: int = Field(ge=1)
    model: str
    items: list[MultimodalEvidenceItem] = Field(default_factory=list, max_length=80)
    rejected_count: int = Field(default=0, ge=0)
    notice: str = (
        "多模态提取结果仅作为候选证据，已保留页码和区域定位；确认前不会写入个人画像。"
    )
