from datetime import datetime, timezone

from app.schemas.profile import ProfileCard
from app.services.pending_ability_catalog import (
    PENDING_ABILITY_TEMPLATES,
    generate_pending_abilities,
)
from app.tasks.catalog import TASK_CATALOG, get_task_definition


EXPECTED_TEMPLATES = {
    "数据驱动判断能力": "能否结合目标、指标与实际影响判断优先级，而不是只按数字大小排序。",
    "模型评测能力": "能否建立合理的评价标准，区分不同失败类型，并判断模型问题来自哪里。",
    "AI场景判断能力": "能否判断一个问题是否真的适合使用AI解决，以及AI应该介入到什么程度。",
    "Agent流程设计能力": "能否把复杂任务拆成清晰步骤，并判断哪些环节适合Agent自动执行、哪些需要用户确认。",
    "技术产品判断能力": "能否在用户需求、技术可行性和产品价值之间进行取舍，而不是只从单一角度判断。",
    "系统性问题诊断能力": "面对失败结果时，能否从数据、模型、流程和产品机制中定位真正的问题来源。",
    "实验验证设计能力": "能否把一个模糊判断转化成可观察、可比较、可验证的实验或任务。",
    "指标设计能力": "能否把抽象目标转化成可衡量的指标，并判断指标是否真的代表目标。",
    "需求拆解能力": "能否把模糊需求拆成具体问题、用户场景、约束条件和可执行任务。",
    "优先级判断能力": "面对多个问题或方案时，能否结合价值、成本、风险和依赖关系进行排序。",
}


def _card(card_id: str, title: str, category: str) -> ProfileCard:
    timestamp = datetime.now(timezone.utc)
    return ProfileCard(
        id=card_id,
        title=title,
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


def test_pending_ability_catalog_exactly_matches_p3_document() -> None:
    actual = {
        template.title: template.description
        for template in PENDING_ABILITY_TEMPLATES
    }

    assert actual == EXPECTED_TEMPLATES


def test_every_task_selects_five_documented_templates_for_user_choice() -> None:
    assert len(TASK_CATALOG) == 12
    for task in TASK_CATALOG.values():
        first = generate_pending_abilities(task, [])
        second = generate_pending_abilities(task, [])

        assert len(first) == 5
        assert first == second
        assert len({item.title for item in first}) == len(first)
        assert all(item.title in EXPECTED_TEMPLATES for item in first)
        assert all(
            item.description == EXPECTED_TEMPLATES[item.title]
            for item in first
        )


def test_representative_tasks_choose_relevant_documented_templates() -> None:
    feature_titles = [
        item.title
        for item in generate_pending_abilities(get_task_definition("F-01"), [])
    ]
    agent_titles = [
        item.title
        for item in generate_pending_abilities(get_task_definition("A-02"), [])
    ]

    assert feature_titles == [
        "需求拆解能力",
        "数据驱动判断能力",
        "Agent流程设计能力",
        "实验验证设计能力",
        "指标设计能力",
    ]
    assert agent_titles == [
        "数据驱动判断能力",
        "模型评测能力",
        "系统性问题诊断能力",
        "AI场景判断能力",
        "技术产品判断能力",
    ]


def test_already_confirmed_template_is_not_offered_again() -> None:
    task = get_task_definition("F-01")
    confirmed = [_card("insight", "需求拆解能力", "洞察分析")]

    generated = generate_pending_abilities(task, confirmed)

    assert len(generated) == 5
    assert "需求拆解能力" not in {item.title for item in generated}


def test_weakly_related_card_does_not_hide_a_pending_gap() -> None:
    task = get_task_definition("F-01")
    weakly_related = [_card("data", "数据复盘能力", "数据驱动")]

    generated = generate_pending_abilities(task, weakly_related)

    assert len(generated) == 5
    assert generated[0].title == "需求拆解能力"
