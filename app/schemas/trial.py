from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


AttributionLayer = Literal[
    "Prompt / 指令层",
    "Model / 基础模型能力",
    "RAG / Retrieval",
    "Tool / 权限与调用",
    "Memory / 长期状态",
    "Workflow / 任务编排",
    "Interaction / UI 与用户控制",
    "Safety / 事实与风险机制",
    "暂无法判断",
]
Confidence = Literal["低", "中", "高"]
EventDecision = Literal["维持", "调整"]
ReflectionChangeType = Literal["新增证据", "加强证据", "冲突证据", "仍待验证"]
TrialEvidenceSource = Literal[
    "ability_card",
    "card_play",
    "answer",
    "material",
    "event",
    "coach",
]
TrialEvidenceKind = Literal["planned", "observed", "deliverable", "reference", "interaction"]
TrialAbilityApplicationStatus = Literal["已应用", "部分应用", "未形成证据"]
TrialVerificationStatus = Literal["accepted", "needs_review", "repaired"]


class A02Metric(BaseModel):
    id: str
    label: str
    current: str
    previous: str


class A02BadCase(BaseModel):
    id: str
    title: str
    description: str


class A02TaskEvent(BaseModel):
    actor: str
    message: str
    instruction: str


class A02CoachPrompt(BaseModel):
    level: str
    title: str
    content: str


class A02RubricCriterion(BaseModel):
    dimension: str
    weight: int
    observable_behavior: str


class A02Task(BaseModel):
    id: Literal["A-02"]
    title: str
    subtitle: str
    role_type: str
    work_stage: str
    primary_skill: str
    supporting_skills: list[str]
    estimated_minutes: str
    difficulty: str
    role: str
    background: str
    goal: str
    metrics: list[A02Metric]
    bad_cases: list[A02BadCase]
    attribution_layers: list[str]
    constraints: list[str]
    event: A02TaskEvent
    coach_prompts: list[A02CoachPrompt]
    rubric: list[A02RubricCriterion]
    source_note: str


class A02Attribution(BaseModel):
    case_id: str
    layer: AttributionLayer
    confidence: Confidence


class A02EvidenceReference(BaseModel):
    source_id: str
    source_type: Literal["case", "metric"]
    explanation: str = Field(default="", max_length=300)


class A02ValidationPlan(BaseModel):
    case_id: str
    action: str = Field(default="", max_length=500)
    expected_signal: str = Field(default="", max_length=300)


class A02Answer(BaseModel):
    attributions: list[A02Attribution] = Field(default_factory=list, max_length=8)
    priority_case_ids: list[str] = Field(default_factory=list, max_length=2)
    evidence: list[A02EvidenceReference] = Field(default_factory=list, max_length=12)
    validation_plans: list[A02ValidationPlan] = Field(default_factory=list, max_length=2)
    event_decision: EventDecision | None = None
    event_priority_case_ids: list[str] = Field(default_factory=list, max_length=2)
    event_reason: str = Field(default="", max_length=500)


class TrialEvidenceItem(BaseModel):
    """A bounded, user-visible evidence item assembled from persisted task data."""

    id: str = Field(min_length=1, max_length=120)
    source: TrialEvidenceSource
    source_id: str = Field(min_length=1, max_length=120)
    kind: TrialEvidenceKind
    label: str = Field(min_length=1, max_length=160)
    content: str = Field(min_length=1, max_length=600)


class TrialAbilityApplication(BaseModel):
    """How one confirmed ability card was used, with references to observed work."""

    card_id: str = Field(min_length=1, max_length=120)
    card_title: str = Field(min_length=1, max_length=120)
    challenge_ids: list[str] = Field(default_factory=list, max_length=3)
    evidence_refs: list[str] = Field(default_factory=list, max_length=8)
    status: TrialAbilityApplicationStatus = "未形成证据"
    basis: str = Field(default="", max_length=500)
    next_step: str = Field(default="", max_length=300)


class TrialEvidenceBundle(BaseModel):
    """Evidence assembled by the server before a model evaluation is persisted."""

    items: list[TrialEvidenceItem] = Field(default_factory=list, max_length=40)
    evidence_refs: list[str] = Field(default_factory=list, max_length=12)
    selected_card_ids: list[str] = Field(default_factory=list, max_length=12)
    ability_applications: list[TrialAbilityApplication] = Field(
        default_factory=list,
        max_length=12,
    )


class TrialDimensionEvaluation(BaseModel):
    dimension: str
    weight: int = Field(default=0, ge=0, le=100)
    score: int = Field(ge=0, le=100)
    evidence: str = Field(max_length=500)
    evidence_refs: list[str] = Field(default_factory=list, max_length=8)


class TrialAbilityEvidence(BaseModel):
    ability: str
    observed_level: Literal["L1", "L2", "L3", "L4", "L5", "证据不足"]
    evidence: str = Field(max_length=600)
    evidence_refs: list[str] = Field(default_factory=list, max_length=8)


class TrialVerification(BaseModel):
    """Deterministic gate applied after a TrialAgent response."""

    status: TrialVerificationStatus = "accepted"
    triggered: bool = False
    reason_codes: list[str] = Field(default_factory=list, max_length=12)
    evidence_coverage: float = Field(default=0.0, ge=0, le=1)
    invalid_evidence_ref_count: int = Field(default=0, ge=0)
    missing_dimension_count: int = Field(default=0, ge=0)
    score_without_evidence_count: int = Field(default=0, ge=0)
    model_reviewed: bool = False
    review_summary: str = Field(default="", max_length=600)


class TrialEvaluation(BaseModel):
    summary: str = Field(max_length=600)
    dimensions: list[TrialDimensionEvaluation] = Field(max_length=8)
    primary_ability: str = ""
    observed_level: Literal["L1", "L2", "L3", "L4", "L5", "证据不足"] = "证据不足"
    level_reason: str = Field(default="", max_length=600)
    supporting_evidence: list[TrialAbilityEvidence] = Field(default_factory=list, max_length=2)
    process_evidence: list[str] = Field(default_factory=list, max_length=6)
    coach_dependency: Literal["独立完成", "轻度提示", "方向性提示", "强提示"] = "独立完成"
    strengths: list[str] = Field(max_length=5)
    gaps: list[str] = Field(max_length=5)
    next_step: str = Field(max_length=300)
    confidence: Literal["低", "中", "高"]
    evidence_refs: list[str] = Field(default_factory=list, max_length=12)
    ability_applications: list[TrialAbilityApplication] = Field(default_factory=list, max_length=12)
    verification: TrialVerification | None = None
    evaluation_protocol: str = "trial-evidence-v1"


class ReflectionChange(BaseModel):
    """One evidence-bound proposal produced after a completed trial."""

    change_type: ReflectionChangeType
    ability: str = Field(min_length=1, max_length=120)
    statement: str = Field(min_length=1, max_length=400)
    evidence_refs: list[str] = Field(min_length=1, max_length=8)
    basis: str = Field(min_length=1, max_length=500)


class ReflectionProposal(BaseModel):
    """A review proposal that cannot directly mutate confirmed profile cards."""

    summary: str = Field(min_length=1, max_length=600)
    changes: list[ReflectionChange] = Field(min_length=1, max_length=6)
    next_verification: str = Field(min_length=1, max_length=300)
    generation_mode: Literal["model", "deterministic_fallback"] = "model"
    profile_update_allowed: Literal[False] = False
    notice: str = "复盘只形成证据变更提案，不会直接修改已确认能力卡。"


class ObservedEvidence(BaseModel):
    task_id: str
    statement: str
    completed_steps: list[str]
    evidence_refs: list[str]
    caveats: list[str]
    evidence_items: list[TrialEvidenceItem] = Field(default_factory=list, max_length=40)
    selected_card_ids: list[str] = Field(default_factory=list, max_length=12)
    primary_ability: str | None = None
    observed_level: Literal["L1", "L2", "L3", "L4", "L5", "证据不足"] | None = None
    level_reason: str | None = None
    confidence: Literal["低", "中", "高"] | None = None
    coach_dependency: Literal["独立完成", "轻度提示", "方向性提示", "强提示"] | None = None
    reflection: ReflectionProposal | None = None


class TrialSession(BaseModel):
    id: str
    task_id: Literal["A-02"]
    status: Literal["in_progress", "submitted"]
    event_revealed: bool
    answer: A02Answer
    created_at: datetime
    updated_at: datetime
    submitted_at: datetime | None = None
    observed_evidence: ObservedEvidence | None = None
    evaluation: TrialEvaluation | None = None


class TrialSessionCreateRequest(BaseModel):
    task_id: Literal["A-02"]


class TrialAnswerUpdateRequest(BaseModel):
    answer: A02Answer
