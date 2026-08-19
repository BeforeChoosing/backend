from collections import Counter

from app.schemas.profile import ProfileCard
from app.schemas.task_catalog import TaskId, TrialTaskCandidate, TrialTaskRecommendation
from app.tasks.catalog import TASK_CATALOG


CATEGORY_SKILL_WEIGHTS: dict[str, dict[str, float]] = {
    "洞察分析": {"用户洞察": 8, "数据驱动": 2, "模型评测": 1},
    "产品策略": {"方案与交互": 6, "商业意识": 5, "AI 产品化": 3},
    "技术落地": {"AI 产品化": 8, "模型评测": 4, "方案与交互": 2},
    "数据驱动": {"数据驱动": 8, "模型评测": 5, "用户洞察": 2},
    "协作沟通": {"跨团队落地": 7, "商业意识": 5, "方案与交互": 2},
    "交互体验": {"方案与交互": 8, "用户洞察": 4, "AI 产品化": 2},
}

TRACK_KEYWORDS: dict[str, tuple[str, ...]] = {
    "feature": ("用户", "体验", "功能", "流程", "交互", "需求", "创作", "留存", "复用"),
    "agent": ("agent", "智能体", "prompt", "rag", "tool", "工作流", "bad case", "归因"),
    "platform": ("平台", "团队", "复用", "基础设施", "开发者", "接口", "治理", "季度"),
    "model": ("模型", "评测", "样本", "指标", "上线", "灰度", "记忆", "memory", "风险"),
}

TASK_KEYWORDS: dict[str, tuple[str, ...]] = {
    "F-01": ("洞察", "创作", "介入", "用户问题"),
    "F-02": ("留存", "复用", "漏斗", "上线复盘"),
    "F-03": ("mvp", "首版", "范围", "验收"),
    "A-01": ("自动化", "边界", "工作流", "确认"),
    "A-02": ("bad case", "失败", "归因", "诊断"),
    "A-03": ("prompt", "rag", "tool", "planner", "技术方案"),
    "P-01": ("多团队", "定制", "平台边界", "共性"),
    "P-02": ("优先级", "季度", "工单", "客户价值"),
    "P-03": ("平台能力", "接口", "memory", "tools", "治理"),
    "M-01": ("上线", "灰度", "模型风险", "分层"),
    "M-02": ("样本", "测试", "评测集", "失败分类"),
    "M-03": ("记忆", "memory", "生命周期", "隐私", "个性化"),
}


def _normalized_text(cards: list[ProfileCard], target_role: str) -> str:
    parts = [target_role]
    for card in cards:
        parts.extend(
            [
                card.title,
                card.category,
                card.description,
                card.detail,
                card.next_verification,
                card.workplace_application,
            ]
        )
    return " ".join(parts).lower()


def _reason_for(
    task_id: str,
    cards: list[ProfileCard],
    skill_scores: Counter[str],
    text: str,
) -> str:
    task = TASK_CATALOG[task_id]
    matching_cards = [
        card.title
        for card in cards
        if CATEGORY_SKILL_WEIGHTS.get(card.category, {}).get(task.primary_skill, 0) > 0
    ][:2]
    matched_keywords = [keyword for keyword in TASK_KEYWORDS[task_id] if keyword in text][:2]
    parts: list[str] = []
    if matching_cards:
        parts.append(f"能力卡“{'、'.join(matching_cards)}”指向{task.primary_skill}")
    elif skill_scores[task.primary_skill] > 0:
        parts.append(f"已确认材料需要继续验证{task.primary_skill}")
    if matched_keywords:
        parts.append(f"待验证描述包含“{'、'.join(matched_keywords)}”")
    if not parts:
        parts.append(f"当前证据对{task.primary_skill}仍不足")
    return "；".join(parts) + "。"


def recommend_trial_task(
    cards: list[ProfileCard],
    completed_task_ids: list[str],
    *,
    target_role: str = "AI 产品经理",
) -> TrialTaskRecommendation:
    """Rank the fixed task catalog without asking the LLM to create or rewrite tasks."""
    if not cards:
        raise ValueError("至少需要一张已确认能力卡。")

    text = _normalized_text(cards, target_role)
    skill_scores: Counter[str] = Counter()
    for card in cards:
        pending_multiplier = 1.2 if card.pending_verification else 1.0
        for skill, weight in CATEGORY_SKILL_WEIGHTS.get(card.category, {}).items():
            skill_scores[skill] += weight * pending_multiplier

    completed = {task_id for task_id in completed_task_ids if task_id in TASK_CATALOG}
    available_ids = [task_id for task_id in TASK_CATALOG if task_id not in completed]
    if not available_ids:
        available_ids = list(TASK_CATALOG)

    ranked: list[tuple[str, float]] = []
    for task_id in available_ids:
        task = TASK_CATALOG[task_id]
        score = 10.0 + skill_scores[task.primary_skill]
        score += sum(skill_scores[skill] * 0.25 for skill in task.supporting_skills)
        score += sum(1.5 for keyword in TRACK_KEYWORDS[task.track] if keyword in text)
        score += sum(3.0 for keyword in TASK_KEYWORDS[task_id] if keyword in text)
        ranked.append((task_id, round(score, 2)))

    order = {task_id: index for index, task_id in enumerate(TASK_CATALOG)}
    ranked.sort(key=lambda item: (-item[1], order[item[0]]))
    candidates = [
        TrialTaskCandidate(
            task_id=task_id,  # type: ignore[arg-type]
            title=TASK_CATALOG[task_id].title,
            primary_skill=TASK_CATALOG[task_id].primary_skill,
            score=score,
            reason=_reason_for(task_id, cards, skill_scores, text),
        )
        for task_id, score in ranked[:3]
    ]
    selected = candidates[0]
    return TrialTaskRecommendation(
        selected_task=TASK_CATALOG[selected.task_id],
        reason=selected.reason,
        candidates=candidates,
        completed_task_ids=sorted(completed),  # type: ignore[arg-type]
    )
