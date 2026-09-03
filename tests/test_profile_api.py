
import pytest

from app.api import profile as profile_api
from app.schemas.profile import (
    AttachmentExperienceCandidate,
    CardProposal,
    MaterialUnderstandingResponse,
)
from app.schemas.profile import ProfileExplorationRequest, ProfileExplorationResponse
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


def test_profile_card_api_lifecycle(tmp_path, monkeypatch, authenticated_client):
    store = ProfileStore(tmp_path / "profile.db")
    monkeypatch.setattr(profile_api, "_profile_store", lambda: store)
    client = authenticated_client

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


def test_profile_overview_returns_submitted_task_evidence(tmp_path, monkeypatch, authenticated_client):
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
    client = authenticated_client

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
            model="qwen-test",
            model_pool="fast",
        )

    def explore_stream(self, request, trace_id, *, on_delta, on_reset=None):
        self.calls += 1
        on_delta('{"reply":"补充你如何')
        on_delta('根据访谈确定范围。","focus_dimension":"decision"}')
        return ProfileExplorationResponse(
            trace_id=trace_id,
            reply="补充你如何根据访谈确定范围。",
            focus_dimension="decision",
            evidence_found=["负责用户访谈"],
            evidence_gap="仍缺少范围取舍的判断依据。",
            potential_hypotheses=["可能具备产品范围判断潜能，仍需验证。"],
            ready_for_proposal=False,
            model="qwen-test",
            model_pool="fast",
        )


class _FakeMaterialUnderstandingAgent:
    def __init__(self):
        self.calls = 0

    async def understand_material(self, request, trace_id):
        self.calls += 1
        return MaterialUnderstandingResponse(
            trace_id=trace_id,
            file_name=request.file_name,
            summary="材料中有一段用户访谈与上线结果，适合继续核对。",
            experience_candidates=[
                AttachmentExperienceCandidate(
                    id="candidate-1",
                    title="校园项目",
                    excerpt="访谈用户并完成上线。",
                    why_worth_exploring="包含行动和结果，可以继续补充判断依据。",
                    suggested_focus="A",
                    source_refs=["material:one"],
                )
            ],
            suggested_action="explore",
            model="qwen3.6-flash",
            model_pool="text:fast",
        )


class _FakeProfileProposalAgent:
    def __init__(self):
        self.calls = 0

    async def propose(self, request, trace_id):
        self.calls += 1
        payload = _card_payload()
        payload["id"] = f"proposal-{self.calls}"
        return profile_api.ProfileProposalResponse(
            trace_id=trace_id,
            experience={
                "title": "校园项目",
                "actions": ["访谈用户", "整理需求"],
                "result": "完成原型验证。",
                "source_refs": ["input:experience_text"],
            },
            card_proposals=[payload],
            next_question="可以继续补充这段经历。",
        )


def test_material_understanding_returns_selectable_experiences_and_caches_result(
    tmp_path, monkeypatch, authenticated_client
):
    store = ProfileStore(tmp_path / "profile.db")
    agent = _FakeMaterialUnderstandingAgent()
    monkeypatch.setattr(profile_api, "_profile_store", lambda: store)
    monkeypatch.setattr(profile_api, "_profile_agent", lambda: agent)
    payload = {
        "file_name": "resume.txt",
        "text": "访谈用户并完成上线。",
        "stored_material_id": "material-one",
    }

    first = authenticated_client.post("/api/v1/profile/materials/understand", json=payload)
    second = authenticated_client.post("/api/v1/profile/materials/understand", json=payload)

    assert first.status_code == 200
    assert first.json()["experience_candidates"][0]["suggested_focus"] == "A"
    assert first.json()["cache_hit"] is False
    assert second.json()["cache_hit"] is True
    assert agent.calls == 1


def test_profile_exploration_reuses_identical_model_result(tmp_path, monkeypatch, authenticated_client):
    store = ProfileStore(tmp_path / "profile.db")
    agent = _FakeProfileExplorationAgent()
    monkeypatch.setattr(profile_api, "_profile_store", lambda: store)
    monkeypatch.setattr(profile_api, "_profile_agent", lambda: agent)
    client = authenticated_client
    request = {
        "experience_text": "我在校园项目中负责访谈用户并整理需求，随后和团队完成了产品原型。",
        "messages": [{"role": "user", "content": "我负责访谈并整理了十五条反馈。"}],
        "request_id": "request-explore-001",
    }

    first = client.post("/api/v1/profile/exploration/messages", json=request)
    second = client.post("/api/v1/profile/exploration/messages", json=request)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["cache_hit"] is False
    assert second.json()["cache_hit"] is True
    assert first.json()["model"] == second.json()["model"] == "qwen-test"
    assert first.json()["reply"] == second.json()["reply"]
    assert agent.calls == 1


def test_profile_proposal_only_reuses_exactly_identical_context(
    tmp_path, monkeypatch, authenticated_client
):
    store = ProfileStore(tmp_path / "profile.db")
    agent = _FakeProfileProposalAgent()
    monkeypatch.setattr(profile_api, "_profile_store", lambda: store)
    monkeypatch.setattr(profile_api, "_profile_agent", lambda: agent)
    base = {
        "experience_text": "我负责访谈用户并整理需求。",
        "target_role": "AI 产品经理",
        "existing_card_titles": ["用户洞察能力"],
    }

    first = authenticated_client.post("/api/v1/profile/proposals", json=base)
    identical = authenticated_client.post("/api/v1/profile/proposals", json=base)
    changed = authenticated_client.post(
        "/api/v1/profile/proposals",
        json={**base, "experience_text": "我负责访谈用户并根据反馈缩小了首版范围。"},
    )

    assert first.status_code == identical.status_code == changed.status_code == 200
    assert first.json()["trace_id"] == identical.json()["trace_id"]
    assert changed.json()["trace_id"] != first.json()["trace_id"]
    assert agent.calls == 2


def test_profile_exploration_server_owns_focus_and_readiness(tmp_path, monkeypatch, authenticated_client):
    store = ProfileStore(tmp_path / "profile.db")
    agent = _FakeProfileExplorationAgent()
    monkeypatch.setattr(profile_api, "_profile_store", lambda: store)
    monkeypatch.setattr(profile_api, "_profile_agent", lambda: agent)
    client = authenticated_client

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


def test_profile_exploration_accepts_a_short_conversational_opening(
    tmp_path, monkeypatch, authenticated_client
):
    """The chat composer promises free-form input, including a brief opening."""
    store = ProfileStore(tmp_path / "profile.db")
    agent = _FakeProfileExplorationAgent()
    monkeypatch.setattr(profile_api, "_profile_store", lambda: store)
    monkeypatch.setattr(profile_api, "_profile_agent", lambda: agent)

    response = authenticated_client.post(
        "/api/v1/profile/exploration/messages",
        json={
            "experience_text": "你是谁",
            "messages": [{"role": "user", "content": "你是谁"}],
            "request_id": "request-short-opening",
        },
    )

    assert response.status_code == 200
    assert response.json()["reply"]
    assert agent.calls == 1


def test_profile_exploration_stream_emits_reply_before_final_payload(
    tmp_path, monkeypatch, authenticated_client
):
    store = ProfileStore(tmp_path / "profile.db")
    agent = _FakeProfileExplorationAgent()
    monkeypatch.setattr(profile_api, "_profile_store", lambda: store)
    monkeypatch.setattr(profile_api, "_profile_agent", lambda: agent)

    response = authenticated_client.post(
        "/api/v1/profile/exploration/messages/stream",
        json={
            "experience_text": "我负责访谈用户并整理需求。",
            "messages": [{"role": "user", "content": "我访谈了十五位用户。"}],
            "request_id": "request-stream-001",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.index("event: delta") < response.text.index("event: done")
    assert '"text":"补充你如何"' in response.text
    assert '"reply":"补充你如何根据访谈确定范围。"' in response.text
    assert '"model":"qwen-test"' in response.text
    assert '"cache_hit":false' in response.text
    assert agent.calls == 1


def test_profile_exploration_accepts_fifty_context_messages():
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"第 {index + 1} 条"}
        for index in range(50)
    ]
    request = ProfileExplorationRequest.model_validate(
        {"experience_text": "测试上下文", "messages": messages}
    )
    assert len(request.messages) == 50


def test_profile_exploration_rejects_more_than_fifty_context_messages():
    messages = [{"role": "user", "content": f"第 {index + 1} 条"} for index in range(51)]
    with pytest.raises(ValueError):
        ProfileExplorationRequest.model_validate(
            {"experience_text": "测试上下文", "messages": messages}
        )
