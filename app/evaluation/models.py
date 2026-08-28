from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.task_catalog import DynamicTrialAnswer, TaskId
from app.schemas.trial import TrialEvaluation


TrialArm = Literal["base_qwen", "prompt_hardened", "sft", "sft_validator"]


class GoldTrialEvaluation(BaseModel):
    """Human-reviewed expectations for one locked evaluation case.

    Scores are deliberately kept as references rather than used to create
    runtime evaluations. A case may specify only the dimensions reviewed by
    the annotators; the remaining dimensions are not silently treated as
    correct.
    """

    dimensions: dict[str, int] = Field(default_factory=dict)
    observed_level: Literal["L1", "L2", "L3", "L4", "L5", "证据不足"] = "证据不足"
    evidence_refs: list[str] = Field(default_factory=list, max_length=40)
    required_ability_applications: list[str] = Field(default_factory=list, max_length=12)


class EvaluationCase(BaseModel):
    """A deterministic input and its human-reviewed reference labels."""

    case_id: str = Field(min_length=1, max_length=120)
    task_id: TaskId
    answer: DynamicTrialAnswer
    gold: GoldTrialEvaluation
    confirmed_card_ids: list[str] = Field(default_factory=list, max_length=12)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CaseEvaluation(BaseModel):
    case_id: str
    task_id: TaskId
    schema_valid: bool
    error_code: str | None = None
    dimension_score_mae: float | None = None
    level_exact_match: bool | None = None
    level_within_one: bool | None = None
    evidence_precision: float | None = None
    evidence_recall: float | None = None
    evidence_f1: float | None = None
    invalid_evidence_ref_count: int = 0
    missing_dimension_count: int = 0
    required_ability_application_recall: float | None = None
    verifier_triggered: bool = False
    api_calls: int = 0
    verifier_api_calls: int = 0
    latency_ms: float | None = None
    notes: list[str] = Field(default_factory=list)


class ArmSummary(BaseModel):
    arm: TrialArm
    case_count: int
    valid_schema_rate: float
    dimension_score_mae: float | None = None
    level_exact_rate: float | None = None
    level_within_one_rate: float | None = None
    evidence_precision: float | None = None
    evidence_recall: float | None = None
    evidence_f1: float | None = None
    invalid_evidence_ref_rate: float | None = None
    verifier_trigger_rate: float | None = None
    mean_api_calls: float | None = None
    mean_latency_ms: float | None = None


class EvaluationReport(BaseModel):
    report_version: str = "trial-eval-v1"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dataset_version: str
    dataset_sha256: str
    model_id: str
    prompt_version: str
    arms: list[ArmSummary] = Field(default_factory=list)
    cases: dict[TrialArm, list[CaseEvaluation]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
