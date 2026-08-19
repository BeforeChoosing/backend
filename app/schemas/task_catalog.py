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
