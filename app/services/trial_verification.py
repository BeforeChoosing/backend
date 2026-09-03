from __future__ import annotations

from collections.abc import Iterable

from app.schemas.task_catalog import DynamicTrialAnswer, TrialTaskDefinition
from app.schemas.trial import (
    TrialEvaluation,
    TrialEvidenceBundle,
    TrialVerification,
)


class TrialVerificationService:
    """Run cheap, evidence-bound checks before an optional model review."""

    SCORE_EVIDENCE_THRESHOLD = 70

    def __init__(self, *, min_evidence_coverage: float = 0.75):
        self.min_evidence_coverage = max(0.0, min(1.0, min_evidence_coverage))

    @staticmethod
    def _evaluation_refs(evaluation: TrialEvaluation) -> list[str]:
        refs = list(evaluation.evidence_refs)
        refs.extend(ref for item in evaluation.dimensions for ref in item.evidence_refs)
        refs.extend(
            ref
            for item in evaluation.supporting_evidence
            for ref in item.evidence_refs
        )
        return list(dict.fromkeys(refs))

    def check(
        self,
        task: TrialTaskDefinition,
        answer: DynamicTrialAnswer,
        evidence_bundle: TrialEvidenceBundle,
        evaluation: TrialEvaluation,
    ) -> TrialVerification:
        del answer  # The bundle is the server-owned representation of the answer.
        reason_codes: list[str] = []
        valid_refs = {item.id for item in evidence_bundle.items}
        returned = {item.dimension: item for item in evaluation.dimensions}
        expected = {item.dimension: item for item in task.rubric}

        missing_dimensions = len(expected.keys() - returned.keys())
        if missing_dimensions:
            reason_codes.append("missing_dimension")
        if len(returned) != len(evaluation.dimensions):
            reason_codes.append("duplicate_dimension")
        if any(
            dimension not in expected
            or item.weight != expected[dimension].weight
            for dimension, item in returned.items()
        ):
            reason_codes.append("rubric_mismatch")

        refs = self._evaluation_refs(evaluation)
        invalid_count = sum(ref not in valid_refs for ref in refs)
        if invalid_count:
            reason_codes.append("invalid_evidence_ref")

        dimensions_with_evidence = 0
        score_without_evidence = 0
        for criterion in task.rubric:
            item = returned.get(criterion.dimension)
            if item is None:
                continue
            valid_dimension_refs = [ref for ref in item.evidence_refs if ref in valid_refs]
            if valid_dimension_refs:
                dimensions_with_evidence += 1
            if item.score >= self.SCORE_EVIDENCE_THRESHOLD and not valid_dimension_refs:
                score_without_evidence += 1
        coverage = (
            dimensions_with_evidence / len(task.rubric)
            if task.rubric
            else 0.0
        )
        if score_without_evidence:
            reason_codes.append("score_without_evidence")
        if coverage < self.min_evidence_coverage:
            reason_codes.append("low_evidence_coverage")
        if evaluation.confidence == "低":
            reason_codes.append("low_confidence")
        if evaluation.observed_level in {"L4", "L5"} and coverage < 0.75:
            reason_codes.append("level_without_evidence")

        # Preserve order while removing duplicate reason codes.
        reason_codes = list(dict.fromkeys(reason_codes))
        triggered = bool(reason_codes)
        return TrialVerification(
            status="needs_review" if triggered else "accepted",
            triggered=triggered,
            reason_codes=reason_codes,
            evidence_coverage=coverage,
            invalid_evidence_ref_count=invalid_count,
            missing_dimension_count=missing_dimensions,
            score_without_evidence_count=score_without_evidence,
        )

    @staticmethod
    def attach(
        evaluation: TrialEvaluation,
        verification: TrialVerification,
    ) -> TrialEvaluation:
        return evaluation.model_copy(update={"verification": verification})


def valid_evidence_refs(bundle: TrialEvidenceBundle) -> Iterable[str]:
    """Expose the server-owned reference set for evaluation tooling."""

    return (item.id for item in bundle.items)
