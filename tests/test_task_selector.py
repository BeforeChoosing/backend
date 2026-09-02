from datetime import datetime, timezone


from app.api import trial as trial_api
from app.schemas.profile import ProfileCard
from app.schemas.profile import ProfileEvidenceRecord
from app.schemas.trial import ObservedEvidence, TrialEvaluation
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


def test_selector_keeps_completed_tasks_available_for_revalidation() -> None:
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
    assert first.selected_task.id in {candidate.task_id for candidate in second.candidates}
    assert first.selected_task.id in second.completed_task_ids


def test_selector_considers_latest_evaluation_gap() -> None:
    card = _card(
        "insight-gap",
        title="用户反馈分析",
        category="洞察分析",
        description="从用户反馈中识别创作流程中的阻塞点",
        next_verification="验证 AI 介入点和用户问题边界",
    )
    evidence = ProfileEvidenceRecord(
        session_id="trial-gap",
        task_id="F-02",
        created_at=datetime.now(timezone.utc),
        observed_evidence=ObservedEvidence(
            task_id="F-02",
            statement="完成 F-02 任务。",
            completed_steps=["定位问题层"],
            evidence_refs=["feedback"],
            caveats=["任务材料为模拟数据。"],
            primary_ability="数据驱动",
            observed_level="L2",
            level_reason="需要补充分层指标与验证停止条件。",
            confidence="低",
            coach_dependency="轻度提示",
        ),
        evaluation=TrialEvaluation(
            summary="需要补充分层指标与验证停止条件。",
            dimensions=[],
            primary_ability="数据驱动",
            observed_level="L2",
            level_reason="证据不足。",
            gaps=["需要补充分层指标与验证停止条件"],
            next_step="补充留存漏斗和分层指标验证",
            strengths=[],
            confidence="低",
        ),
    )

    recommendation = recommend_trial_task([card], ["F-02"], evidence_records=[evidence])

    assert recommendation.selected_task.id == "F-01"
    assert "洞察" in recommendation.reason or "创作" in recommendation.reason


def test_catalog_and_recommendation_api_use_confirmed_cards(tmp_path, monkeypatch, authenticated_client) -> None:
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
    client = authenticated_client

    catalog = client.get("/api/v1/trial/catalog")
    recommendation = client.post(
        "/api/v1/trial/recommendations",
        json={"selected_card_ids": ["platform"]},
    )

    assert catalog.status_code == 200
    assert len(catalog.json()) == 12
    assert recommendation.status_code == 200
    assert recommendation.json()["selected_task"]["id"] == "P-01"
