from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.api import career as career_api
from app.main import app
from app.knowledge.retriever import KnowledgeChunk
from app.schemas.career import CareerRecommendation
from app.schemas.profile import CardProposal
from app.services.profile_store import ProfileStore


def _card_payload(card_id: str = "career-card-1") -> dict:
    return CardProposal(
        id=card_id,
        title="用户研究",
        category="洞察分析",
        description="通过访谈和反馈整理识别用户问题",
        detail="从用户反馈中提炼可行动问题，并形成需求假设。",
        icon="Eye",
        color_tone="purple",
        claim_level="interpretation",
        evidence_type="self_report",
        evidence_quote="我访谈用户并根据反馈调整了方案",
        source_refs=["input:experience_text"],
        pending_verification=True,
        next_verification="补充样本选择和决策变化",
        match_reason="依据用户自述的访谈和方案调整",
        workplace_application="支持 AI 功能需求分析",
    ).model_dump(mode="json")


class _FakeRetriever:
    def search(self, *args, **kwargs):
        return [
            KnowledgeChunk(
                id="chk-career-1",
                document_id="job-ai-product-manager-v1",
                document_title="AI 产品经理 · 岗位能力参考库",
                corpus="career",
                content="AI 产品经理需要把用户问题转化为可验证的产品方案。",
                heading_path=("AI 产品经理", "用户研究"),
                source_locator="jobs/ai_product_manager.md#用户研究",
                trust_level="secondary_summary",
                source_note="测试资料",
                score=2.0,
            )
        ]


class _FakeCareerAgent:
    async def recommend(self, cards, retrieved, next_task, next_task_reason):
        return CareerRecommendation(
            summary=f"已基于 {cards[0].title} 形成 AI 产品经理探索建议。",
            supported=[
                {
                    "claim": "用户研究能力可支持需求分析。",
                    "card_ids": [cards[0].id],
                    "citation_ids": [retrieved[0].id],
                }
            ],
            unknowns=["尚未验证 Bad Case 归因能力。"],
            next_task_id=next_task.id,
            next_task_title=next_task.title,
            next_task_reason=next_task_reason,
            confidence="中",
            citations=[
                {
                    "id": retrieved[0].id,
                    "document_title": retrieved[0].document_title,
                    "source_locator": retrieved[0].source_locator,
                    "content": retrieved[0].content,
                    "trust_level": retrieved[0].trust_level,
                    "source_note": retrieved[0].source_note,
                }
            ],
        )


def test_career_recommendation_uses_confirmed_cards_and_citations(tmp_path, monkeypatch):
    store = ProfileStore(tmp_path / "profile.db")
    store.confirm_cards([CardProposal.model_validate(_card_payload())], trace_id="trace-career")
    monkeypatch.setattr(career_api, "_profile_store", lambda: store)
    monkeypatch.setattr(career_api, "_knowledge_retriever", lambda: _FakeRetriever())
    monkeypatch.setattr(career_api, "_career_agent", lambda: _FakeCareerAgent())

    response = TestClient(app).post(
        "/api/v1/career/recommendations",
        json={"selected_card_ids": ["career-card-1"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["role_id"] == "ai_product_manager"
    assert payload["next_task_id"] == "F-01"
    assert payload["next_task_title"] == "AI 到底应该进入创作链路的哪一步？"
    assert payload["supported"][0]["citation_ids"] == ["chk-career-1"]
    assert payload["citations"][0]["source_locator"].startswith("jobs/")


def test_career_recommendation_rejects_unconfirmed_card(tmp_path, monkeypatch):
    monkeypatch.setattr(career_api, "_profile_store", lambda: ProfileStore(tmp_path / "profile.db"))

    response = TestClient(app).post(
        "/api/v1/career/recommendations",
        json={"selected_card_ids": ["not-confirmed"]},
    )

    assert response.status_code == 422
