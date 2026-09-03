from types import SimpleNamespace

from app.knowledge.query_planner import build_career_queries


def _card(**overrides):
    values = {
        "title": "用户痛点洞察",
        "category": "洞察分析",
        "description": "通过访谈和反馈整理识别用户问题",
        "detail": "从用户反馈中提炼可行动问题，并形成需求假设。",
        "next_verification": "补充样本选择和决策变化",
        "workplace_application": "支持 AI 功能需求分析",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_career_query_planner_keeps_role_and_card_intents_separate():
    queries = build_career_queries(
        [_card(), _card(title="数据验证与复盘", category="数据驱动")],
        target_role="AI 产品经理",
    )

    assert queries[0] == "AI 产品经理 岗位职责 能力要求 工作内容"
    assert len(queries) == 3
    assert "用户痛点洞察" in queries[1]
    assert "数据验证与复盘" in queries[2]
    assert all(len(query) <= 430 for query in queries)


def test_career_query_planner_deduplicates_repeated_card_queries():
    queries = build_career_queries(
        [_card(), _card()],
        target_role="AI 产品经理",
    )

    assert len(queries) == 2
