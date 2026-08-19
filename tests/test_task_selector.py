from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.api import trial as trial_api
from app.main import app
from app.schemas.profile import ProfileCard
from app.services.profile_store import ProfileStore
from app.services.task_selector import recommend_trial_task


def _card(
    card_id: str,
    *,
    title: str,
    category: str,
    description: str,
    next_verification: str,
) -> ProfileCard:
    now = datetime.now(timezone.utc)
    return ProfileCard.model_validate(
        {
            "id": card_id,
            "title": title,
            "category": category,
            "description": description,
            "detail": description,
            "icon": "Sparkles",
            "color_tone": "emerald",
            "claim_level": "interpretation",
            "evidence_type": "self_report",
            "evidence_quote": description,
            "source_refs": ["test"],
            "pending_verification": True,
            "next_verification": next_verification,
            "match_reason": "测试",
            "workplace_application": description,
            "status": "confirmed",
            "created_at": now,
            "updated_at": now,
        }
    )


def test_selector_uses_card_evidence_instead_of_fixed_task() -> None:
    insight = _card(
        "insight",
        title="新用户访谈与流失定位",
        category="洞察分析",
        description="结合用户反馈定位 AI 功能首次使用的阻塞点",
        next_verification="验证新用户为什么没有形成复用",
    )
    evaluation = _card(
        "evaluation",
        title="小样本模型评测",
        category="数据驱动",
        description="用有限样本识别模型高风险错误",
        next_verification="设计 100 条样本的评测集和一票否决规则",
    )

    insight_result = recommend_trial_task([insight], [])
    evaluation_result = recommend_trial_task([evaluation], [])

    assert insight_result.selected_task.id == "F-01"
    assert evaluation_result.selected_task.id == "M-02"
    assert insight_result.selected_task.id != evaluation_result.selected_task.id


def test_selector_skips_tasks_with_recorded_evidence() -> None:
    card = _card(
        "memory",
        title="用户记忆策略",
        category="数据驱动",
        description="分析 Agent 长期记忆的用户价值和隐私边界",
        next_verification="验证 memory 生命周期与删除逻辑",
    )

    first = recommend_trial_task([card], [])
    second = recommend_trial_task([card], [first.selected_task.id])

    assert first.selected_task.id == "M-03"
    assert second.selected_task.id != first.selected_task.id
    assert first.selected_task.id in second.completed_task_ids


def test_catalog_and_recommendation_api_use_confirmed_cards(tmp_path, monkeypatch) -> None:
    store = ProfileStore(tmp_path / "profile.db")
    card = _card(
        "platform",
        title="多团队平台需求治理",
        category="协作沟通",
        description="协调多个团队并抽象平台共性和定制边界",
        next_verification="验证平台边界和首个试点客户",
    )
    store.confirm_cards([card], trace_id="trace-selector")
    monkeypatch.setattr(trial_api, "_profile_store", lambda: store)
    client = TestClient(app)

    catalog = client.get("/api/v1/trial/catalog")
    recommendation = client.post(
        "/api/v1/trial/recommendations",
        json={"selected_card_ids": ["platform"]},
    )

    assert catalog.status_code == 200
    assert len(catalog.json()) == 12
    assert recommendation.status_code == 200
    assert recommendation.json()["selected_task"]["id"] == "P-01"
