from dataclasses import dataclass

from app.schemas.profile import ProfileCard
from app.schemas.task_catalog import (
    DynamicTrialPendingAbility,
    TrialAbilityChallenge,
    TrialTaskDefinition,
)
@dataclass(frozen=True)
class PendingAbilityTemplate:
    title: str
    description: str
    skill_scores: dict[str, int]
    keywords: tuple[str, ...]


# P3 技术说明规定的完整模板库。待验证能力只能从这里选择，不允许按技能名临时造卡。
PENDING_ABILITY_TEMPLATES: tuple[PendingAbilityTemplate, ...] = (
    PendingAbilityTemplate(
        "数据驱动判断能力",
        "能否结合目标、指标与实际影响判断优先级，而不是只按数字大小排序。",
        {"数据驱动": 100, "优先级判断": 90, "商业意识": 55},
        ("数据", "指标", "优先级", "影响", "排序", "漏斗"),
    ),
    PendingAbilityTemplate(
        "模型评测能力",
        "能否建立合理的评价标准，区分不同失败类型，并判断模型问题来自哪里。",
        {"模型评测": 100, "数据驱动": 45},
        ("模型", "评测", "评价", "失败类型", "效果", "上线"),
    ),
    PendingAbilityTemplate(
        "AI场景判断能力",
        "能否判断一个问题是否真的适合使用AI解决，以及AI应该介入到什么程度。",
        {"AI产品化": 100, "商业意识": 55, "用户洞察": 35},
        ("AI", "场景", "介入", "产品定义", "功能判断", "机会"),
    ),
    PendingAbilityTemplate(
        "Agent流程设计能力",
        "能否把复杂任务拆成清晰步骤，并判断哪些环节适合Agent自动执行、哪些需要用户确认。",
        {"方案与交互": 100, "AI产品化": 90, "跨团队落地": 60},
        ("Agent", "流程", "自动", "确认", "步骤", "工作流", "回退"),
    ),
    PendingAbilityTemplate(
        "技术产品判断能力",
        "能否在用户需求、技术可行性和产品价值之间进行取舍，而不是只从单一角度判断。",
        {"AI产品化": 95, "商业意识": 90, "方案与交互": 65, "跨团队落地": 55},
        ("技术", "可行性", "产品价值", "取舍", "成本", "维护"),
    ),
    PendingAbilityTemplate(
        "系统性问题诊断能力",
        "面对失败结果时，能否从数据、模型、流程和产品机制中定位真正的问题来源。",
        {"模型评测": 95, "数据驱动": 75, "AI产品化": 70},
        ("失败", "诊断", "归因", "问题来源", "数据", "模型", "流程"),
    ),
    PendingAbilityTemplate(
        "实验验证设计能力",
        "能否把一个模糊判断转化成可观察、可比较、可验证的实验或任务。",
        {"数据驱动": 92, "模型评测": 85, "方案与交互": 55},
        ("实验", "验证", "测试", "假设", "可观察", "可比较"),
    ),
    PendingAbilityTemplate(
        "指标设计能力",
        "能否把抽象目标转化成可衡量的指标，并判断指标是否真的代表目标。",
        {"数据驱动": 90, "模型评测": 65, "商业意识": 55},
        ("指标", "衡量", "目标", "平均", "通过标准", "门槛"),
    ),
    PendingAbilityTemplate(
        "需求拆解能力",
        "能否把模糊需求拆成具体问题、用户场景、约束条件和可执行任务。",
        {"用户洞察": 100, "方案与交互": 82, "跨团队落地": 45},
        ("需求", "用户", "场景", "约束", "具体问题", "任务"),
    ),
    PendingAbilityTemplate(
        "优先级判断能力",
        "面对多个问题或方案时，能否结合价值、成本、风险和依赖关系进行排序。",
        {"优先级判断": 100, "商业意识": 92, "数据驱动": 88, "跨团队落地": 65},
        ("优先级", "价值", "成本", "风险", "依赖", "排序", "重排"),
    ),
)

PENDING_ABILITY_TITLES = frozenset(template.title for template in PENDING_ABILITY_TEMPLATES)


def _challenge_text(challenge: TrialAbilityChallenge) -> str:
    return " ".join((
        challenge.title,
        challenge.scenario,
        challenge.prompt,
        challenge.reference_behavior,
        *challenge.target_skills,
    )).lower()


def _template_score(
    template: PendingAbilityTemplate,
    challenge: TrialAbilityChallenge,
) -> int:
    base = max(
        (template.skill_scores.get(skill, 0) for skill in challenge.target_skills),
        default=0,
    )
    text = _challenge_text(challenge)
    keyword_score = sum(3 for keyword in template.keywords if keyword.lower() in text)
    return base + keyword_score


def _template_already_confirmed(
    template: PendingAbilityTemplate,
    confirmed_cards: list[ProfileCard],
) -> bool:
    core_title = template.title.removesuffix("能力")
    return any(
        template.title == card.title
        or core_title in card.title
        for card in confirmed_cards
    )


def generate_pending_abilities(
    task: TrialTaskDefinition,
    confirmed_cards: list[ProfileCard],
) -> list[DynamicTrialPendingAbility]:
    """Select five task-relevant candidates from the documented template catalog."""

    ranked: list[tuple[int, int, PendingAbilityTemplate, TrialAbilityChallenge]] = []
    for template_index, template in enumerate(PENDING_ABILITY_TEMPLATES):
        if _template_already_confirmed(template, confirmed_cards):
            continue
        scored_challenges = [
            (_template_score(template, challenge), challenge)
            for challenge in task.ability_challenges
        ]
        score, best_challenge = max(scored_challenges, key=lambda item: item[0])
        if score > 0:
            ranked.append((score, -template_index, template, best_challenge))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [
        DynamicTrialPendingAbility(
            id=f"pending:{task.id}:{index}",
            challenge_id=challenge.id,
            title=template.title,
            description=template.description,
            target_skills=challenge.target_skills,
        )
        for index, (_, _, template, challenge) in enumerate(ranked[:5], start=1)
    ]
