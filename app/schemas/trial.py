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


class ObservedEvidence(BaseModel):
    task_id: str
    statement: str
    completed_steps: list[str]
    evidence_refs: list[str]
    caveats: list[str]


class TrialDimensionEvaluation(BaseModel):
    dimension: str
    score: int = Field(ge=0, le=100)
    evidence: str = Field(max_length=500)


class TrialEvaluation(BaseModel):
    summary: str = Field(max_length=600)
    dimensions: list[TrialDimensionEvaluation] = Field(max_length=8)
    strengths: list[str] = Field(max_length=5)
    gaps: list[str] = Field(max_length=5)
    next_step: str = Field(max_length=300)
    confidence: Literal["低", "中", "高"]


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
