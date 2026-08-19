from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


CardCategory = Literal[
    "洞察分析",
    "产品策略",
    "技术落地",
    "数据驱动",
    "协作沟通",
    "交互体验",
]
ClaimLevel = Literal["fact", "interpretation", "hypothesis"]
EvidenceType = Literal["documented_fact", "self_report", "inference"]


class ProfileProposalRequest(BaseModel):
    experience_text: str = Field(min_length=20, max_length=12000)
    target_role: str | None = Field(default=None, max_length=120)
    existing_card_titles: list[str] = Field(default_factory=list, max_length=20)


class ExperienceSummary(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    actions: list[str] = Field(default_factory=list, max_length=8)
    result: str | None = Field(default=None, max_length=500)
    source_refs: list[str] = Field(default_factory=list, max_length=10)


class CardProposal(BaseModel):
    id: str
    title: str = Field(min_length=1, max_length=80)
    category: CardCategory
    description: str = Field(min_length=1, max_length=240)
    detail: str = Field(min_length=1, max_length=600)
    icon: str = "Sparkles"
    color_tone: Literal["purple", "blue", "emerald", "amber", "rose"] = "emerald"
    claim_level: ClaimLevel = "interpretation"
    evidence_type: EvidenceType = "self_report"
    evidence_quote: str = Field(min_length=1, max_length=500)
    source_refs: list[str] = Field(default_factory=list, max_length=10)
    pending_verification: bool = True
    next_verification: str = Field(min_length=1, max_length=240)
    match_reason: str = Field(min_length=1, max_length=300)
    workplace_application: str = Field(min_length=1, max_length=300)


class ProfileProposalResponse(BaseModel):
    trace_id: str
    experience: ExperienceSummary
    card_proposals: list[CardProposal] = Field(max_length=5)
    next_question: str = Field(min_length=1, max_length=300)
    notice: str = "这些内容是候选证据，确认前不会写入你的长期画像。"


class ProfileCard(CardProposal):
    status: Literal["confirmed"] = "confirmed"
    source_trace_id: str | None = Field(default=None, max_length=100)
    created_at: datetime
    updated_at: datetime


class ConfirmProfileCardsRequest(BaseModel):
    cards: list[CardProposal] = Field(min_length=1, max_length=20)
    trace_id: str | None = Field(default=None, max_length=100)


class ProfileCardPatchRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=80)
    description: str | None = Field(default=None, min_length=1, max_length=240)
    detail: str | None = Field(default=None, min_length=1, max_length=600)
    workplace_application: str | None = Field(default=None, min_length=1, max_length=300)


class ProfileCardsResponse(BaseModel):
    version: int
    updated_at: datetime | None = None
    cards: list[ProfileCard] = Field(default_factory=list)
    notice: str = "这些是用户确认后的个人画像卡片。"
