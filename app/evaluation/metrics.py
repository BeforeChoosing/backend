from __future__ import annotations

from collections.abc import Iterable
from math import sqrt
from typing import Any

from pydantic import ValidationError

from app.evaluation.models import CaseEvaluation, EvaluationCase
from app.schemas.trial import TrialEvaluation


_LEVEL_ORDER = {"证据不足": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}


def _all_evidence_refs(evaluation: TrialEvaluation) -> list[str]:
    refs: list[str] = []
    refs.extend(evaluation.evidence_refs)
    for dimension in evaluation.dimensions:
        refs.extend(dimension.evidence_refs)
    for item in evaluation.supporting_evidence:
        refs.extend(item.evidence_refs)
    return list(dict.fromkeys(refs))


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _schema_error_code(error: Exception) -> str:
    if isinstance(error, ValidationError):
        return "invalid_schema"
    return "invalid_prediction"


def evaluate_case(
    case: EvaluationCase,
    prediction: TrialEvaluation | dict[str, Any],
    *,
    valid_evidence_refs: Iterable[str],
    api_calls: int = 0,
    verifier_api_calls: int = 0,
    verifier_triggered: bool = False,
    latency_ms: float | None = None,
) -> CaseEvaluation:
    """Score one prediction without raising on malformed model output."""

    try:
        evaluation = (
            prediction
            if isinstance(prediction, TrialEvaluation)
            else TrialEvaluation.model_validate(prediction)
        )
    except (ValidationError, TypeError, ValueError) as exc:
        return CaseEvaluation(
            case_id=case.case_id,
            task_id=case.task_id,
            schema_valid=False,
            error_code=_schema_error_code(exc),
            api_calls=api_calls,
            verifier_api_calls=verifier_api_calls,
            verifier_triggered=verifier_triggered,
            latency_ms=latency_ms,
        )

    valid_refs = set(valid_evidence_refs)
    predicted_refs = _all_evidence_refs(evaluation)
    gold_refs = set(case.gold.evidence_refs)
    grounded_refs = [ref for ref in predicted_refs if ref in valid_refs]
    invalid_count = sum(ref not in valid_refs for ref in predicted_refs)
    overlap = set(grounded_refs) & gold_refs
    precision = len(grounded_refs) / len(predicted_refs) if predicted_refs else 0.0
    recall = len(overlap) / len(gold_refs) if gold_refs else 1.0

    predicted_scores = {item.dimension: item.score for item in evaluation.dimensions}
    errors = [
        abs(predicted_scores.get(dimension, 0) - score)
        for dimension, score in case.gold.dimensions.items()
    ]
    missing_dimensions = sum(
        dimension not in predicted_scores for dimension in case.gold.dimensions
    )
    level_value = _LEVEL_ORDER.get(evaluation.observed_level, 0)
    gold_level_value = _LEVEL_ORDER.get(case.gold.observed_level, 0)
    required_cards = set(case.gold.required_ability_applications)
    returned_cards = {item.card_id for item in evaluation.ability_applications}
    card_recall = (
        len(required_cards & returned_cards) / len(required_cards)
        if required_cards
        else None
    )
    return CaseEvaluation(
        case_id=case.case_id,
        task_id=case.task_id,
        schema_valid=True,
        dimension_score_mae=(sum(errors) / len(errors) if errors else None),
        level_exact_match=evaluation.observed_level == case.gold.observed_level,
        level_within_one=abs(level_value - gold_level_value) <= 1,
        evidence_precision=precision,
        evidence_recall=recall,
        evidence_f1=_f1(precision, recall),
        invalid_evidence_ref_count=invalid_count,
        missing_dimension_count=missing_dimensions,
        required_ability_application_recall=card_recall,
        verifier_triggered=verifier_triggered,
        api_calls=api_calls,
        verifier_api_calls=verifier_api_calls,
        latency_ms=latency_ms,
    )


def summarize_cases(
    arm: str,
    cases: list[CaseEvaluation],
) -> dict[str, Any]:
    """Return JSON-friendly aggregate metrics for report generation."""

    count = len(cases)
    valid = [case for case in cases if case.schema_valid]

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "arm": arm,
        "case_count": count,
        "valid_schema_rate": len(valid) / count if count else 0.0,
        "dimension_score_mae": mean(
            [case.dimension_score_mae for case in valid if case.dimension_score_mae is not None]
        ),
        "level_exact_rate": mean(
            [1.0 if case.level_exact_match else 0.0 for case in valid if case.level_exact_match is not None]
        ),
        "level_within_one_rate": mean(
            [1.0 if case.level_within_one else 0.0 for case in valid if case.level_within_one is not None]
        ),
        "evidence_precision": mean(
            [case.evidence_precision for case in valid if case.evidence_precision is not None]
        ),
        "evidence_recall": mean(
            [case.evidence_recall for case in valid if case.evidence_recall is not None]
        ),
        "evidence_f1": mean(
            [case.evidence_f1 for case in valid if case.evidence_f1 is not None]
        ),
        "invalid_evidence_ref_rate": (
            sum(case.invalid_evidence_ref_count for case in valid) / len(valid)
            if valid
            else None
        ),
        "verifier_trigger_rate": mean(
            [1.0 if case.verifier_triggered else 0.0 for case in cases]
        ),
        "mean_api_calls": mean([float(case.api_calls) for case in cases]),
        "mean_latency_ms": mean(
            [case.latency_ms for case in cases if case.latency_ms is not None]
        ),
    }
