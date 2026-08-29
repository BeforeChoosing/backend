from fastapi.testclient import TestClient

from app.api import profile as profile_api
from app.main import app
from app.schemas.profile import CardProposal
from app.schemas.profile import ProfileExplorationResponse
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


class _FakeProfileExplorationAgent:
    def __init__(self):
        self.calls = 0

    async def explore(self, request, trace_id):
        self.calls += 1
        return ProfileExplorationResponse(
            trace_id=trace_id,
            reply="补充你如何根据访谈内容确定第一版范围。",
            focus_dimension="decision",
            evidence_found=["负责用户访谈"],
            evidence_gap="仍缺少范围取舍的判断依据。",
            potential_hypotheses=["可能具备产品范围判断潜能，仍需验证。"],
            ready_for_proposal=False,
        )


def test_profile_exploration_reuses_identical_model_result(tmp_path, monkeypatch):
    store = ProfileStore(tmp_path / "profile.db")
    agent = _FakeProfileExplorationAgent()
    monkeypatch.setattr(profile_api, "_profile_store", lambda: store)
    monkeypatch.setattr(profile_api, "_profile_agent", lambda: agent)
    client = TestClient(app)
    request = {
        "experience_text": "我在校园项目中负责访谈用户并整理需求，随后和团队完成了产品原型。",
        "messages": [{"role": "user", "content": "我负责访谈并整理了十五条反馈。"}],
        "request_id": "request-explore-001",
    }

    first = client.post("/api/v1/profile/exploration/messages", json=request)
    second = client.post("/api/v1/profile/exploration/messages", json=request)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert agent.calls == 1


def test_profile_exploration_server_owns_focus_and_readiness(tmp_path, monkeypatch):
    store = ProfileStore(tmp_path / "profile.db")
    agent = _FakeProfileExplorationAgent()
    monkeypatch.setattr(profile_api, "_profile_store", lambda: store)
    monkeypatch.setattr(profile_api, "_profile_agent", lambda: agent)
    client = TestClient(app)

    response = client.post(
        "/api/v1/profile/exploration/messages",
        json={
            "experience_text": "我在校园项目中负责整理需求并完成一版可用原型。",
            "request_id": "request-explore-controller",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["focus_dimension"] == "decision"
    assert payload["ready_for_proposal"] is False
    assert payload["coverage"]["decision"] == "missing"
