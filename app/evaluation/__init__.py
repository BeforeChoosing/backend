"""Offline evaluation contracts for TrialAgent experiments."""

from app.evaluation.metrics import evaluate_case
from app.evaluation.models import (
    CaseEvaluation,
    EvaluationCase,
    EvaluationReport,
    GoldTrialEvaluation,
    TrialArm,
)

__all__ = [
    "CaseEvaluation",
    "EvaluationCase",
    "EvaluationReport",
    "GoldTrialEvaluation",
    "TrialArm",
    "evaluate_case",
]
