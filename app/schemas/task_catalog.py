from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


TaskId = Literal[
    "F-01", "F-02", "F-03",
    "A-01", "A-02", "A-03",
    "P-01", "P-02", "P-03",
    "M-01", "M-02", "M-03",
]


class TrialTaskStep(BaseModel):
    id: str
    title: str
    input_mode: str
    instruction: str
    constraint: str


class TrialTaskEvent(BaseModel):
    actor: str
    message: str
    instruction: str


class TrialTaskRubricCriterion(BaseModel):
    dimension: str
    weight: int = Field(ge=0, le=100)
    observable_behavior: str


class TrialTaskDefinition(BaseModel):
    id: TaskId
    track: Literal["feature", "agent", "platform", "model"]
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
    constraints: list[str]
    steps: list[TrialTaskStep] = Field(min_length=5, max_length=5)
    event: TrialTaskEvent
    coach_prompts: list[str] = Field(min_length=3, max_length=3)
    rubric: list[TrialTaskRubricCriterion]
    level_anchors: dict[Literal["L1", "L2", "L3", "L4", "L5"], str] = Field(default_factory=dict)
    source_note: str = "任务结构来自 CoachAgent AI 产品经理职业试路任务库 v2.0；业务数字与案例为模拟试路材料。"


class TrialTaskCandidate(BaseModel):
    task_id: TaskId
    title: str
    primary_skill: str
    score: float
    reason: str


class TrialTaskRecommendationRequest(BaseModel):
    selected_card_ids: list[str] = Field(default_factory=list, max_length=4)
    target_role: str = Field(default="AI 产品经理", max_length=120)


class TrialTaskRecommendation(BaseModel):
    selected_task: TrialTaskDefinition
    reason: str
    candidates: list[TrialTaskCandidate] = Field(max_length=3)
    completed_task_ids: list[TaskId] = Field(default_factory=list)
    selection_policy: str = "基于已确认能力卡、待验证项、职业方向与已完成任务进行确定性排序；Qwen 不生成或改写任务。"


class DynamicTrialAnswer(BaseModel):
    step_answers: dict[str, str] = Field(default_factory=dict)
    viewed_material_ids: list[str] = Field(default_factory=list, max_length=30)
    evidence_refs: list[str] = Field(default_factory=list, max_length=12)
    step_revisions: dict[str, int] = Field(default_factory=dict)
    coach_usage: list["DynamicTrialCoachUsage"] = Field(default_factory=list, max_length=20)
    event_decision: Literal["维持", "调整"] | None = None
    event_response: str = Field(default="", max_length=1200)


class DynamicTrialCoachUsage(BaseModel):
    level: Literal[1, 2, 3]
    prompt: str = Field(max_length=500)
    used_at: datetime


class DynamicTrialCoachRequest(BaseModel):
    level: Literal[1, 2, 3]


class DynamicTrialCoachResponse(BaseModel):
    prompt: str
    usage: DynamicTrialCoachUsage


class DynamicTrialSession(BaseModel):
    id: str
    task_id: TaskId
    status: Literal["in_progress", "submitted"]
    event_revealed: bool
    answer: DynamicTrialAnswer
    created_at: datetime
    updated_at: datetime
    submitted_at: datetime | None = None
    observed_evidence: "ObservedEvidence | None" = None
    evaluation: "TrialEvaluation | None" = None


class DynamicTrialSessionCreateRequest(BaseModel):
    task_id: TaskId


class DynamicTrialAnswerUpdateRequest(BaseModel):
    answer: DynamicTrialAnswer


from app.schemas.trial import ObservedEvidence, TrialEvaluation  # noqa: E402

DynamicTrialSession.model_rebuild()
DynamicTrialAnswer.model_rebuild()
