from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.trial import ObservedEvidence, TrialEvaluation


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
ExplorationFocus = Literal[
    "ownership",
    "decision",
    "constraint",
    "collaboration",
    "result",
    "transfer",
    "evidence",
]
ExplorationCoverageStatus = Literal["missing", "weak", "sufficient", "confirmed"]
ModelTier = Literal["fast", "balanced", "reasoning"]


class ProfileConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=1000)


class ProfileExplorationRequest(BaseModel):
    # Exploration is a conversation: a user may begin with a brief greeting or
    # question before sharing a full experience.
    experience_text: str = Field(min_length=1, max_length=12000)
    messages: list[ProfileConversationMessage] = Field(default_factory=list, max_length=50)
    # The client may echo the dimensions already prompted by the server.  This
    # is optional so older clients remain compatible; coverage is still
    # recomputed from the full user transcript on every request.
    focus_history: list[ExplorationFocus] = Field(default_factory=list, max_length=50)
    target_role: str | None = Field(default=None, max_length=120)
    existing_card_titles: list[str] = Field(default_factory=list, max_length=20)
    request_id: str | None = Field(default=None, min_length=8, max_length=100)
    model_tier: ModelTier = "fast"


class ProfileExplorationResponse(BaseModel):
    trace_id: str
    reply: str = Field(min_length=1, max_length=300)
    focus_dimension: ExplorationFocus
    evidence_found: list[str] = Field(default_factory=list, max_length=5)
    evidence_gap: str = Field(min_length=1, max_length=300)
    potential_hypotheses: list[str] = Field(default_factory=list, max_length=3)
    ready_for_proposal: bool = False
    coverage: dict[ExplorationFocus, ExplorationCoverageStatus] = Field(default_factory=dict)
    model: str | None = Field(default=None, max_length=120)
    model_pool: str | None = Field(default=None, max_length=120)
    cache_hit: bool = False
    notice: str = "潜能线索仅用于继续补充经历，确认前不会写入个人画像。"


class ProfileConversationSnapshotMessage(BaseModel):
    id: str = Field(min_length=1, max_length=120)
    role: Literal["user", "ai"]
    content: str = Field(min_length=1, max_length=1000)
    timestamp: str = Field(default="", max_length=40)
    detected_signals: list[str] = Field(default_factory=list, max_length=5)
    model: str | None = Field(default=None, max_length=120)
    cache_hit: bool | None = None


class ProfileConversationMaterial(BaseModel):
    name: str = Field(min_length=1, max_length=240)
    size: str = Field(default="", max_length=40)
    type: Literal["resume", "portfolio", "link"]


class ProfileConversationSnapshotUpsert(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    messages: list[ProfileConversationSnapshotMessage] = Field(min_length=1, max_length=60)
    evidence: str = Field(default="", max_length=12000)
    materials: list[ProfileConversationMaterial] = Field(default_factory=list, max_length=20)
    target_career_state: Literal["unselected", "has_target", "no_target"] = "unselected"
    target_role: str = Field(default="", max_length=120)
    model_tier: ModelTier = "balanced"


class ProfileConversationSnapshot(ProfileConversationSnapshotUpsert):
    id: str = Field(min_length=1, max_length=120)
    created_at: datetime
    updated_at: datetime


class ProfileProposalRequest(BaseModel):
    experience_text: str = Field(min_length=1, max_length=12000)
    target_role: str | None = Field(default=None, max_length=120)
    existing_card_titles: list[str] = Field(default_factory=list, max_length=20)


class MaterialExtractResponse(BaseModel):
    file_name: str
    text: str = Field(min_length=1, max_length=12000)
    char_count: int = Field(ge=1)
    truncated: bool = False
    notice: str = "仅提取文档中的可复制文本；内容需由用户核对，且尚未写入长期画像。"


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


class ProfileEvidenceRecord(BaseModel):
    """A task-level evidence record written after a submitted trial."""

    session_id: str
    task_id: str
    created_at: datetime
    observed_evidence: ObservedEvidence
    evaluation: TrialEvaluation | None = None


class ProfileOverviewResponse(ProfileCardsResponse):
    """The data consumed by the profile screen and the next-task selector."""

    evidence: list[ProfileEvidenceRecord] = Field(default_factory=list)
    completed_task_ids: list[str] = Field(default_factory=list)
    notice: str = "能力卡来自用户确认，任务证据来自已提交的试路评价。"
