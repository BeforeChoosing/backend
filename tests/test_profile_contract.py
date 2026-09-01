import asyncio

from app.agents.profile_agent import ProfileAgent
from app.schemas.profile import ProfileExplorationRequest, ProfileProposalRequest


class FakeGateway:
    class settings:
        qwen_fast_model = "qwen3.6-flash"

    def generate_json(self, system_prompt: str, user_prompt: str, **kwargs) -> dict:
        return {
            "experience": {
                "title": "校园项目",
                "actions": ["访谈用户", "调整方案"],
                "result": "完成可用原型",
                "source_refs": ["input:experience_text"],
            },
            "card_proposals": [
                {
                    "title": "用户研究",
                    "category": "洞察分析",
                    "description": "通过访谈和反馈整理识别用户问题",
                    "detail": "从用户反馈中提炼可行动问题。",
                    "claim_level": "interpretation",
                    "evidence_type": "self_report",
                    "evidence_quote": "访谈用户并根据反馈调整方案",
                    "source_refs": ["input:experience_text"],
                    "pending_verification": True,
                    "next_verification": "补充样本选择和决策变化",
                    "match_reason": "依据用户自述的访谈和方案调整",
                    "workplace_application": "支持 AI 功能需求分析",
                }
            ],
            "next_question": "你本人具体完成了哪一步？",
        }


def test_profile_agent_returns_pending_cards():
    response = asyncio.run(ProfileAgent(FakeGateway()).propose(
        ProfileProposalRequest(experience_text="我在校园项目中访谈用户并根据反馈调整了方案，最后完成了可用原型。"),
        "trace-test-1234",
    ))
    assert response.card_proposals[0].title == "用户研究"
    assert response.card_proposals[0].pending_verification is True
    assert response.card_proposals[0].source_refs == ["input:experience_text"]


def test_profile_requests_accept_any_non_empty_user_input():
    assert ProfileExplorationRequest(experience_text="你是谁").experience_text == "你是谁"
    assert ProfileProposalRequest(experience_text="你是谁").experience_text == "你是谁"


class InventedQuoteGateway(FakeGateway):
    def generate_json(self, system_prompt: str, user_prompt: str, **kwargs) -> dict:
        payload = super().generate_json(system_prompt, user_prompt, **kwargs)
        payload["card_proposals"][0]["claim_level"] = "fact"
        payload["card_proposals"][0]["evidence_type"] = "documented_fact"
        payload["card_proposals"][0]["evidence_quote"] = "获得全国一等奖"
        return payload


def test_profile_agent_downgrades_unverifiable_quote():
    response = asyncio.run(ProfileAgent(InventedQuoteGateway()).propose(
        ProfileProposalRequest(experience_text="我在校园项目中访谈用户并调整了方案，最后完成了可用原型。"),
        "trace-test-invalid-quote",
    ))

    card = response.card_proposals[0]
    assert card.claim_level == "hypothesis"
    assert card.evidence_type == "inference"
    assert card.evidence_quote == "模型未返回可逐字核对的原文片段"


class ExplorationGateway:
    class settings:
        qwen_fast_model = "qwen3.6-flash"

    def generate_json(self, system_prompt: str, user_prompt: str, **kwargs) -> dict:
        assert kwargs["model"] == "qwen3.6-flash"
        assert "不重复此前 assistant 已经给出的引导" in system_prompt
        assert "先用一个短句回应用户刚刚说清的具体行动或结果" in system_prompt
        assert "不像审核表或访谈提纲" in system_prompt
        assert "校园项目" in user_prompt
        assert "负责访谈" in user_prompt
        return {
            "reply": "补充你如何从访谈记录中确定优先解决的问题，以及放弃了哪些备选方向。",
            "focus_dimension": "decision",
            "evidence_found": ["用户明确负责访谈"],
            "evidence_gap": "仍缺少方案取舍的判断依据。",
            "potential_hypotheses": ["可能具备基于证据进行产品取舍的潜能，仍需验证。"],
            "ready_for_proposal": False,
        }


def test_profile_agent_exploration_returns_one_evidence_bound_focus():
    response = asyncio.run(ProfileAgent(ExplorationGateway()).explore(
        ProfileExplorationRequest(
            experience_text="我在校园项目中负责访谈用户并整理需求，随后和团队完成了产品原型。",
            messages=[{"role": "user", "content": "我负责访谈，并整理了十五条反馈。"}],
        ),
        "trace-explore-1234",
    ))

    assert response.focus_dimension == "decision"
    assert response.evidence_found == ["用户明确负责访谈"]
    assert response.ready_for_proposal is False


class QuestionExplorationGateway(ExplorationGateway):
    def generate_json(self, system_prompt: str, user_prompt: str, **kwargs) -> dict:
        payload = super().generate_json(system_prompt, user_prompt, **kwargs)
        payload["reply"] = "你具体做了什么？"
        payload["focus_dimension"] = "unknown"
        return payload


def test_profile_agent_exploration_normalizes_invalid_prompt_shape():
    response = asyncio.run(ProfileAgent(QuestionExplorationGateway()).explore(
        ProfileExplorationRequest(
            experience_text="我在校园项目中负责访谈用户并整理需求，随后和团队完成了产品原型。",
        ),
        "trace-explore-invalid",
    ))

    assert response.focus_dimension == "ownership"
    assert "？" not in response.reply
