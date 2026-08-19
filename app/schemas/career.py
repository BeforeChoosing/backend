from typing import Literal

from pydantic import BaseModel, Field


class CareerRecommendationRequest(BaseModel):
    selected_card_ids: list[str] = Field(min_length=1, max_length=4)
    target_role: Literal["AI 产品经理"] = "AI 产品经理"


class CareerCitation(BaseModel):
    id: str
    document_title: str
    source_locator: str
    content: str = Field(max_length=1200)
    trust_level: str
    source_note: str


class CareerSupport(BaseModel):
    claim: str = Field(min_length=1, max_length=300)
    card_ids: list[str] = Field(default_factory=list, max_length=4)
    citation_ids: list[str] = Field(default_factory=list, max_length=5)


class CareerRecommendation(BaseModel):
    role_id: Literal["ai_product_manager"] = "ai_product_manager"
    role_title: str = "AI 产品经理"
    summary: str = Field(min_length=1, max_length=500)
    supported: list[CareerSupport] = Field(default_factory=list, max_length=6)
    unknowns: list[str] = Field(default_factory=list, max_length=6)
    next_task_id: Literal["A-02"] = "A-02"
    confidence: Literal["低", "中", "高"]
    citations: list[CareerCitation] = Field(default_factory=list, max_length=5)
    notice: str = "推荐依据来自已确认能力卡和本地岗位知识片段，不等同于录用或胜任力结论。"
