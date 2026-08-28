from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from pydantic import ValidationError

from app.agents.trial_agent import TrialAgent
from app.schemas.task_catalog import DynamicTrialAnswer, TrialTaskDefinition
from app.schemas.trial import TrialEvaluation


@dataclass(frozen=True)
class ValidationResult:
    status: str
    confidence: float
    schema_valid: bool
    evidence_coverage: float
    invalid_evidence_refs: tuple[str, ...] = ()
    missing_dimensions: tuple[str, ...] = ()
    extra_dimensions: tuple[str, ...] = ()
    duplicate_dimensions: tuple[str, ...] = ()
    weight_mismatch_dimensions: tuple[str, ...] = ()
    score_without_evidence_count: int = 0
    reason_codes: tuple[str, ...] = ()
    evaluation: dict[str, Any] | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "confidence": round(self.confidence, 4),
            "schema_valid": self.schema_valid,
            "evidence_coverage": round(self.evidence_coverage, 4),
            "invalid_evidence_refs": list(self.invalid_evidence_refs),
            "missing_dimensions": list(self.missing_dimensions),
            "extra_dimensions": list(self.extra_dimensions),
            "duplicate_dimensions": list(self.duplicate_dimensions),
            "weight_mismatch_dimensions": list(self.weight_mismatch_dimensions),
            "score_without_evidence_count": self.score_without_evidence_count,
            "reason_codes": list(self.reason_codes),
            "details": dict(self.details),
        }


def _as_ref_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _all_refs(raw: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    refs.extend(_as_ref_list(raw.get("evidence_refs")))
    dimensions = raw.get("dimensions")
    if isinstance(dimensions, list):
        for item in dimensions:
            if isinstance(item, dict):
                refs.extend(_as_ref_list(item.get("evidence_refs")))
    supporting = raw.get("supporting_evidence")
    if isinstance(supporting, list):
        for item in supporting:
            if isinstance(item, dict):
                refs.extend(_as_ref_list(item.get("evidence_refs")))
    applications = raw.get("ability_applications")
    if isinstance(applications, list):
        for item in applications:
            if isinstance(item, dict):
                refs.extend(_as_ref_list(item.get("evidence_refs")))
    return list(dict.fromkeys(refs))


def _raw_dimension_names(raw: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    dimensions = raw.get("dimensions")
    if not isinstance(dimensions, list):
        return [], []
    names: list[str] = []
    rows: list[dict[str, Any]] = []
    for item in dimensions:
        if not isinstance(item, dict):
            continue
        name = item.get("dimension")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
            rows.append(item)
    return names, rows


def _weight_mismatches(
    expected: dict[str, int],
    rows: list[dict[str, Any]],
) -> tuple[str, ...]:
    mismatches: list[str] = []
    seen: set[str] = set()
    for row in rows:
        dimension = row.get("dimension")
        if not isinstance(dimension, str) or dimension in seen or dimension not in expected:
            continue
        seen.add(dimension)
        if row.get("weight") != expected[dimension]:
            mismatches.append(dimension)
    return tuple(mismatches)


def _confidence_score(
    *,
    schema_valid: bool,
    missing_count: int,
    extra_count: int,
    invalid_ref_count: int,
    score_without_evidence_count: int,
    evidence_coverage: float,
    observed_level: str | None,
    all_ref_count: int,
    model_confidence: str | None,
) -> float:
    if not schema_valid:
        return 0.0
    score = 1.0
    score -= min(0.25, missing_count * 0.08)
    score -= min(0.12, extra_count * 0.06)
    score -= min(0.3, invalid_ref_count * 0.12)
    score -= min(0.25, score_without_evidence_count * 0.12)
    if all_ref_count == 0:
        score -= 0.12
    elif evidence_coverage < 1:
        score -= (1 - evidence_coverage) * 0.18
    if observed_level in {"L3", "L4", "L5"} and all_ref_count == 0:
        score -= 0.12
    score = min(score, {"高": 1.0, "中": 0.74, "低": 0.5}.get(model_confidence, 0.0))
    return max(0.0, min(1.0, score))


def validate_evaluation(
    task: TrialTaskDefinition,
    answer: DynamicTrialAnswer,
    raw: dict[str, Any],
    *,
    valid_evidence_refs: Iterable[str],
) -> ValidationResult:
    """Validate a teacher result without making another model call."""

    raw_evaluation = raw.get("evaluation") if isinstance(raw.get("evaluation"), dict) else raw
    if not isinstance(raw_evaluation, dict):
        return ValidationResult(
            status="rejected",
            confidence=0.0,
            schema_valid=False,
            evidence_coverage=0.0,
            reason_codes=("schema_invalid", "top_level_not_object"),
        )

    expected = [criterion.dimension for criterion in task.rubric]
    returned_names, raw_dimensions = _raw_dimension_names(raw_evaluation)
    returned_set = set(returned_names)
    expected_set = set(expected)
    missing = tuple(name for name in expected if name not in returned_set)
    extra = tuple(name for name in returned_names if name not in expected_set)
    duplicate_dimensions = tuple(
        name for name in dict.fromkeys(returned_names) if returned_names.count(name) > 1
    )
    weight_mismatches = _weight_mismatches(
        {criterion.dimension: criterion.weight for criterion in task.rubric},
        raw_dimensions,
    )
    valid_refs = {ref for ref in valid_evidence_refs if isinstance(ref, str) and ref.strip()}
    all_refs = _all_refs(raw_evaluation)
    invalid_refs = tuple(ref for ref in all_refs if ref not in valid_refs)
    used_valid_refs = [ref for ref in all_refs if ref in valid_refs]
    evidence_coverage = len(set(used_valid_refs)) / len(set(all_refs)) if all_refs else 0.0
    score_without_evidence = sum(
        1
        for row in raw_dimensions
        if isinstance(row.get("score"), (int, float))
        and row.get("score", 0) >= 60
        and not _as_ref_list(row.get("evidence_refs"))
    )
    reason_codes: list[str] = []
    if missing:
        reason_codes.append("missing_dimensions")
    if extra:
        reason_codes.append("extra_dimensions")
    if duplicate_dimensions:
        reason_codes.append("duplicate_dimensions")
    if weight_mismatches:
        reason_codes.append("weight_mismatch")
    if invalid_refs:
        reason_codes.append("invalid_evidence_ref")
    if score_without_evidence:
        reason_codes.append("score_without_evidence")
    if not all_refs:
        reason_codes.append("no_evidence_reference")

    try:
        normalized = TrialAgent._normalize_dynamic(task, answer, raw_evaluation)
    except (ValidationError, ValueError, TypeError) as exc:
        return ValidationResult(
            status="rejected",
            confidence=0.0,
            schema_valid=False,
            evidence_coverage=evidence_coverage,
            invalid_evidence_refs=invalid_refs,
            missing_dimensions=missing,
            extra_dimensions=extra,
            duplicate_dimensions=duplicate_dimensions,
            weight_mismatch_dimensions=weight_mismatches,
            score_without_evidence_count=score_without_evidence,
            reason_codes=tuple(dict.fromkeys(["schema_invalid", *reason_codes])),
            details={"error": str(exc)[:500]},
        )

    confidence = _confidence_score(
        schema_valid=True,
        missing_count=len(missing),
        extra_count=len(extra),
        invalid_ref_count=len(invalid_refs),
        score_without_evidence_count=score_without_evidence,
        evidence_coverage=evidence_coverage,
        observed_level=normalized.observed_level,
        all_ref_count=len(all_refs),
        model_confidence=normalized.confidence,
    )
    if normalized.confidence != "高":
        reason_codes.append("model_confidence_not_high")
    status = "silver_auto" if not reason_codes and confidence >= 0.85 else "needs_review"
    return ValidationResult(
        status=status,
        confidence=confidence,
        schema_valid=True,
        evidence_coverage=evidence_coverage,
        invalid_evidence_refs=invalid_refs,
        missing_dimensions=missing,
        extra_dimensions=extra,
        duplicate_dimensions=duplicate_dimensions,
        weight_mismatch_dimensions=weight_mismatches,
        score_without_evidence_count=score_without_evidence,
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        evaluation=normalized.model_dump(mode="json"),
        details={
            "expected_dimensions": expected,
            "returned_dimensions": returned_names,
            "valid_evidence_ref_count": len(valid_refs),
            "answer_step_count": len([value for value in answer.step_answers.values() if value.strip()]),
        },
    )
