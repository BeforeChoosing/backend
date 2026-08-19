from fastapi.testclient import TestClient

from app.api import profile as profile_api
from app.main import app
from app.schemas.profile import CardProposal
from app.services.profile_store import ProfileStore


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
