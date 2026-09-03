from app.schemas.profile import ProfileCard
from app.schemas.task_catalog import (
    DynamicTrialCardPlayRound,
    DynamicTrialPendingAbility,
    TrialAbilityChallenge,
)


CATEGORY_SKILL_WEIGHTS: dict[str, dict[str, float]] = {
    "洞察分析": {
        "用户洞察": 8,
        "数据驱动": 2,
        "模型评测": 1,
        "商业意识": 1,
    },
    "产品策略": {
        "方案与交互": 7,
        "商业意识": 5,
        "AI产品化": 4,
        "优先级判断": 8,
        "跨团队落地": 2,
    },
    "技术落地": {
        "AI产品化": 8,
        "模型评测": 5,
        "方案与交互": 2,
        "创新趋势": 7,
        "跨团队落地": 2,
    },
    "数据驱动": {
        "数据驱动": 8,
        "模型评测": 6,
        "用户洞察": 2,
        "商业意识": 3,
        "优先级判断": 5,
    },
    "协作沟通": {
        "跨团队落地": 8,
        "商业意识": 5,
        "方案与交互": 2,
        "优先级判断": 2,
    },
    "交互体验": {
        "方案与交互": 8,
        "用户洞察": 5,
        "AI产品化": 2,
        "跨团队落地": 1,
    },
}


def card_skill_score(card: ProfileCard, target_skills: list[str]) -> float:
    weights = CATEGORY_SKILL_WEIGHTS.get(card.category, {})
    return max((weights.get(skill, 0) for skill in target_skills), default=0)


def evaluate_card_play_round(
    challenge: TrialAbilityChallenge,
    selected_cards: list[ProfileCard],
    pending_abilities: list[DynamicTrialPendingAbility] | None = None,
) -> DynamicTrialCardPlayRound:
    pending_abilities = pending_abilities or []
    if not selected_cards and not pending_abilities:
        raise ValueError("每个能力应用挑战至少需要选择一张能力卡。")
    if len(selected_cards) + len(pending_abilities) > challenge.max_cards:
        raise ValueError(f"每个能力应用挑战最多选择 {challenge.max_cards} 张能力卡。")

    scored_cards = [
        (card, card_skill_score(card, challenge.target_skills))
        for card in selected_cards
    ]
    matched_cards = [card for card, score in scored_cards if score > 0]
    matched_skills = [
        skill
        for skill in challenge.target_skills
        if any(
            CATEGORY_SKILL_WEIGHTS.get(card.category, {}).get(skill, 0) > 0
            for card in selected_cards
        )
    ]
    direct_match = any(score >= 6 for _, score in scored_cards)
    pending_matched_skills = [
        skill
        for skill in challenge.target_skills
        if any(skill in ability.target_skills for ability in pending_abilities)
    ]
    matched_skills = list(dict.fromkeys([*matched_skills, *pending_matched_skills]))
    pending_ids = [ability.id for ability in pending_abilities]

    skill_text = "、".join(challenge.target_skills)
    if pending_abilities and not selected_cards:
        match_level = "partial"
        feedback = (
            f"现有能力卡与“{skill_text}”没有明确对应。"
            f"本轮将把“{'、'.join(ability.title for ability in pending_abilities)}”作为待验证能力，"
            "通过任务表现补充新的行为证据。"
        )
    elif pending_abilities:
        match_level = "partial"
        feedback = (
            "当前选择的已有能力可以支持部分任务，但仍不足以覆盖核心要求。"
            f"本轮将同时验证“{'、'.join(ability.title for ability in pending_abilities)}”。"
        )
    elif direct_match or len(matched_cards) >= 2:
        match_level = "high"
        feedback = (
            f"所选能力与“{skill_text}”直接对应，可用于本轮任务要求。"
            f"参考表现：{challenge.reference_behavior}"
        )
    elif matched_cards:
        match_level = "partial"
        feedback = (
            f"所选能力可提供辅助支持，但与“{skill_text}”的直接对应仍不充分。"
            f"参考表现：{challenge.reference_behavior}"
        )
    else:
        match_level = "low"
        feedback = (
            f"当前选择与“{skill_text}”关联较弱。"
            f"本轮参考表现：{challenge.reference_behavior}"
        )

    return DynamicTrialCardPlayRound(
        challenge_id=challenge.id,
        selected_card_ids=[*[card.id for card in selected_cards], *pending_ids],
        match_level=match_level,
        matched_card_ids=[*[card.id for card in matched_cards], *pending_ids],
        matched_skills=matched_skills,
        feedback=feedback,
    )
