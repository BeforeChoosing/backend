from datetime import datetime, timezone

from app.schemas.profile import ProfileCard
from app.services.ability_matching import evaluate_card_play_round
from app.tasks.catalog import TASK_CATALOG, get_task_definition


def _card(card_id: str, category: str) -> ProfileCard:
    timestamp = datetime.now(timezone.utc)
    return ProfileCard(
        id=card_id,
        title=f"{category}能力",
        category=category,  # type: ignore[arg-type]
        description="来自用户确认经历的能力描述。",
        detail="该能力已经由用户确认，可用于试路任务中的能力应用判断。",
        evidence_quote="来自用户确认经历的能力描述。",
        source_refs=["input:experience_text"],
        next_verification="在试路任务中继续验证。",
        match_reason="与目标岗位存在关联。",
        workplace_application="用于完成岗位任务中的判断与交付。",
        created_at=timestamp,
        updated_at=timestamp,
    )


def test_all_tasks_expose_three_material_based_ability_challenges() -> None:
    assert len(TASK_CATALOG) == 12
    for task in TASK_CATALOG.values():
        assert len(task.ability_challenges) == 3
        assert len({item.id for item in task.ability_challenges}) == 3
        assert [item.target_skills[0] for item in task.ability_challenges] == [
            item.dimension for item in task.rubric[:3]
        ]
        assert all(
            item.reference_behavior in item.scenario
            for item in task.ability_challenges
        )


def test_card_play_match_has_high_partial_and_low_results() -> None:
    challenge = get_task_definition("F-01").ability_challenges[0]

    high = evaluate_card_play_round(challenge, [_card("insight", "洞察分析")])
    partial = evaluate_card_play_round(challenge, [_card("data", "数据驱动")])
    low = evaluate_card_play_round(challenge, [_card("strategy", "产品策略")])

    assert high.match_level == "high"
    assert high.matched_card_ids == ["insight"]
    assert partial.match_level == "partial"
    assert partial.matched_card_ids == ["data"]
    assert low.match_level == "low"
    assert low.matched_card_ids == []
