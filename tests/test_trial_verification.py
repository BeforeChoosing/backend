from app.schemas.task_catalog import DynamicTrialAnswer
from app.schemas.trial import TrialDimensionEvaluation, TrialEvaluation
from app.services.trial_scoring import TrialScoringService
from app.services.trial_verification import TrialVerificationService
from app.tasks.catalog import get_task_definition


def test_verifier_accepts_grounded_complete_evaluation() -> None:
    task = get_task_definition("F-01")
    answer = DynamicTrialAnswer(
        step_answers={step.id: "完成了这一步并记录了依据。" for step in task.steps},
        event_decision="调整",
        event_response="根据研发约束收缩范围。",
    )
    bundle = TrialScoringService.build_evidence(task, answer, [])
    evaluation = TrialEvaluation(
        summary="覆盖任务中的判断与调整。",
        dimensions=[
            TrialDimensionEvaluation(
                dimension=item.dimension,
                weight=item.weight,
                score=70,
                evidence="关联了作答和事件依据。",
                evidence_refs=["answer:problem", "event:decision"],
            )
            for item in task.rubric
        ],
        primary_ability=task.primary_skill,
        observed_level="L3",
        level_reason="覆盖关键作答步骤。",
        confidence="中",
        strengths=["有明确取舍"],
        gaps=["继续补充结果证据"],
        next_step="在下一任务记录结果。",
        evidence_refs=["answer:problem", "event:decision"],
    )

    result = TrialVerificationService().check(task, answer, bundle, evaluation)

    assert result.status == "accepted"
    assert result.triggered is False
    assert result.evidence_coverage == 1.0


def test_verifier_routes_low_confidence_and_invalid_refs_to_review() -> None:
    task = get_task_definition("F-01")
    answer = DynamicTrialAnswer(step_answers={})
    bundle = TrialScoringService.build_evidence(task, answer, [])
    evaluation = TrialEvaluation(
        summary="证据不足。",
        dimensions=[
            TrialDimensionEvaluation(
                dimension=task.rubric[0].dimension,
                score=96,
                evidence="没有对应作答。",
                evidence_refs=["invented:ref"],
            )
        ],
        observed_level="L5",
        confidence="高",
        strengths=[],
        gaps=[],
        next_step="补充作答。",
    )

    result = TrialVerificationService().check(task, answer, bundle, evaluation)

    assert result.status == "needs_review"
    assert result.triggered is True
    assert "invalid_evidence_ref" in result.reason_codes
    assert "missing_dimension" in result.reason_codes
    assert result.invalid_evidence_ref_count == 1
