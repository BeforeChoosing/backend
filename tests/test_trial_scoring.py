from app.schemas.profile import CardProposal
from app.schemas.task_catalog import DynamicTrialAnswer, DynamicTrialCardPlayRound
from app.schemas.trial import TrialDimensionEvaluation, TrialEvaluation
from app.services.trial_scoring import TrialScoringService
from app.tasks.catalog import get_task_definition


def _card() -> CardProposal:
    return CardProposal(
        id="card-insight",
        title="用户问题拆解",
        category="洞察分析",
        description="从反馈和行为中定位关键问题。",
        detail="能够把混杂反馈整理成可验证的问题判断。",
        evidence_quote="通过访谈和行为数据定位主要阻塞点。",
        source_refs=["input:experience"],
        next_verification="在试路任务中验证问题定义。",
        match_reason="与任务中的用户洞察要求相关。",
        workplace_application="用于定义首版产品问题。",
    )


def test_dynamic_evidence_binds_cards_answers_materials_and_event() -> None:
    task = get_task_definition("F-01")
    answer = DynamicTrialAnswer(
        selected_card_ids=["card-insight"],
        card_play_rounds=[
            DynamicTrialCardPlayRound(
                challenge_id=challenge.id,
                selected_card_ids=["card-insight"],
                match_level="high",
                matched_card_ids=["card-insight"],
                matched_skills=challenge.target_skills,
                feedback="能力与本轮要求直接对应。",
            )
            for challenge in task.ability_challenges
        ],
        card_play_completed=True,
        step_answers={step.id: f"完成{step.title}，给出具体判断和验证动作。" for step in task.steps},
        viewed_material_ids=[task.materials[0].id],
        evidence_refs=[task.materials[0].id],
        event_decision="调整",
        event_response="根据新增约束缩小首版范围并调整验证顺序。",
    )

    bundle = TrialScoringService.build_evidence(task, answer, [_card()])
    item_ids = {item.id for item in bundle.items}

    assert "card:card-insight" in item_ids
    assert "card_play:F-01-C01:card-insight" in item_ids
    assert "answer:problem" in item_ids
    assert f"material:{task.materials[0].id}" in item_ids
    assert "event:decision" in item_ids
    assert bundle.ability_applications[0].status == "已应用"
    assert "answer:problem" in bundle.ability_applications[0].evidence_refs


def test_scoring_cannot_keep_high_score_without_observed_answers() -> None:
    task = get_task_definition("F-01")
    answer = DynamicTrialAnswer(
        selected_card_ids=["card-insight"],
        card_play_completed=True,
        event_decision="调整",
        event_response="只保留必要范围。",
    )
    model_evaluation = TrialEvaluation(
        summary="模型给出了高分。",
        dimensions=[
            TrialDimensionEvaluation(
                dimension=criterion.dimension,
                weight=criterion.weight,
                score=98,
                evidence="模型声称证据充分。",
                evidence_refs=["invented:evidence"],
            )
            for criterion in task.rubric
        ],
        primary_ability=task.primary_skill,
        observed_level="L5",
        level_reason="模型声称达到最高等级。",
        strengths=["模型声称完成"],
        gaps=[],
        next_step="继续任务。",
        confidence="高",
    )

    evaluation, _ = TrialScoringService.finalize_dynamic(
        task,
        answer,
        [_card()],
        model_evaluation,
    )

    assert evaluation.observed_level == "证据不足"
    assert all(item.score <= 35 for item in evaluation.dimensions)
    assert all(not ref.startswith("invented:") for item in evaluation.dimensions for ref in item.evidence_refs)
