from app.agents.reflection_agent import ReflectionAgent
from app.schemas.task_catalog import DynamicTrialAnswer
from app.schemas.trial import TrialDimensionEvaluation, TrialEvaluation
from app.tasks.catalog import get_task_definition


def _evaluation(primary_ability: str) -> TrialEvaluation:
    return TrialEvaluation(
        summary="完成了任务并在事件后调整。",
        dimensions=[
            TrialDimensionEvaluation(
                dimension="方案与交互",
                weight=50,
                score=82,
                evidence="在事件后缩小了首版范围。",
            )
        ],
        primary_ability=primary_ability,
        observed_level="L3",
        level_reason="能根据新限制调整方案。",
        strengths=["响应了事件"],
        gaps=["仍缺少跨任务证据"],
        next_step="在另一项任务中继续验证事件响应。",
        confidence="中",
    )


def test_reflection_filters_invented_references_and_abilities() -> None:
    task = get_task_definition("F-03")
    evaluation = _evaluation(task.primary_skill)
    references = [
        {"reference_id": "answer:scope", "content": "收缩首版范围"},
        {"reference_id": "evaluation:level", "content": "L3"},
        {"reference_id": "card_play:hypothesis", "content": "任务前假设"},
    ]
    raw = {
        "summary": "本次形成了新的任务证据。",
        "changes": [
            {
                "change_type": "新增证据",
                "ability": "模型自创能力",
                "statement": "事件后调整了范围。",
                "evidence_refs": ["answer:scope", "invented:ref"],
                "basis": "作答中明确收缩了范围。",
            },
            {
                "change_type": "加强证据",
                "ability": task.primary_skill,
                "statement": "任务前假设不能单独成为能力证据。",
                "evidence_refs": ["card_play:hypothesis"],
                "basis": "只有任务前预期。",
            },
        ],
        "next_verification": "换一个任务继续验证。",
    }

    reflection = ReflectionAgent._normalize(raw, task, evaluation, references)

    assert len(reflection.changes) == 1
    assert reflection.changes[0].ability == task.primary_skill
    assert reflection.changes[0].evidence_refs == ["answer:scope"]
    assert reflection.profile_update_allowed is False


def test_reflection_fallback_is_explicit_and_conservative() -> None:
    task = get_task_definition("F-03")
    evaluation = _evaluation(task.primary_skill)
    answer = DynamicTrialAnswer(
        step_answers={step.id: "已完成" for step in task.steps},
        event_decision="调整",
        event_response="事件后缩小范围。",
    )

    reflection = ReflectionAgent.fallback(task, answer, evaluation, [], [])

    assert reflection.generation_mode == "deterministic_fallback"
    assert reflection.changes[0].change_type == "仍待验证"
    assert reflection.profile_update_allowed is False
