from fastapi.testclient import TestClient

from app.api import profile as profile_api
from app.main import app
from app.schemas.profile import CardProposal
from app.services.profile_store import ProfileStore
from app.schemas.trial import ObservedEvidence, TrialEvaluation, TrialDimensionEvaluation


def _card_payload() -> dict:
    return CardProposal(
        id="card-api-1",
        title="用户研究",
        category="洞察分析",
        description="通过访谈和反馈整理识别用户问题",
        detail="从用户反馈中提炼可行动问题。",
        icon="Eye",
        color_tone="purple",
        claim_level="interpretation",
        evidence_type="self_report",
        evidence_quote="访谈用户并根据反馈调整方案",
        source_refs=["input:experience_text"],
        pending_verification=True,
        next_verification="补充样本选择和决策变化",
        match_reason="依据用户自述的访谈和方案调整",
        workplace_application="支持 AI 功能需求分析",
    ).model_dump(mode="json")


def test_profile_card_api_lifecycle(tmp_path, monkeypatch):
    store = ProfileStore(tmp_path / "profile.db")
    monkeypatch.setattr(profile_api, "_profile_store", lambda: store)
    client = TestClient(app)

    confirmed = client.post(
        "/api/v1/profile/cards/confirm",
        json={"cards": [_card_payload()], "trace_id": "trace-api"},
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["cards"][0]["status"] == "confirmed"

    updated = client.patch(
        "/api/v1/profile/cards/card-api-1",
        json={"title": "用户洞察"},
    )
    assert updated.status_code == 200
    assert updated.json()["cards"][0]["title"] == "用户洞察"

    deleted = client.delete("/api/v1/profile/cards/card-api-1")
    assert deleted.status_code == 200
    assert deleted.json()["cards"] == []

    missing = client.delete("/api/v1/profile/cards/card-api-1")
    assert missing.status_code == 404


def test_profile_overview_returns_submitted_task_evidence(tmp_path, monkeypatch):
    store = ProfileStore(tmp_path / "profile.db")
    store.record_observed_evidence(
        "dynamic-trial-test",
        ObservedEvidence(
            task_id="F-01",
            statement="完成 F-01 试路任务并形成用户洞察证据。",
            completed_steps=["判断问题类型"],
            evidence_refs=["material-feedback"],
            caveats=["任务材料为模拟数据。"],
            primary_ability="用户洞察",
            observed_level="L3",
            level_reason="能够引用材料并形成单一问题判断。",
            confidence="中",
            coach_dependency="独立完成",
        ),
        TrialEvaluation(
            summary="能够从反馈中界定核心问题。",
            dimensions=[TrialDimensionEvaluation(dimension="用户洞察", weight=40, score=72, evidence="引用反馈形成判断。")],
            primary_ability="用户洞察",
            observed_level="L3",
            level_reason="判断有材料支持。",
            strengths=["问题边界清晰。"],
            gaps=["需要补充验证停止条件。"],
            next_step="补充验证停止条件。",
            confidence="中",
        ),
    )
    monkeypatch.setattr(profile_api, "_profile_store", lambda: store)
    client = TestClient(app)

    overview = client.get("/api/v1/profile/overview")

    assert overview.status_code == 200
    assert overview.json()["completed_task_ids"] == ["F-01"]
    assert overview.json()["evidence"][0]["evaluation"]["observed_level"] == "L3"
