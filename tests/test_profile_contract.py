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
        ProfileProposalRequest(
            experience_text="我在校园项目中访谈用户并根据反馈调整了方案，最后完成了可用原型。",
            experience_id="conversation-1",
        ),
        "trace-test-1234",
    ))
    assert response.card_proposals[0].title == "用户研究能力"
    assert response.card_proposals[0].pending_verification is True
    assert response.card_proposals[0].experience_id == "conversation-1"
    assert response.card_proposals[0].source_refs[0] == "experience:conversation-1"
    assert response.card_proposals[0].evidence_history[0].experience_id == "conversation-1"


def test_profile_agent_rejects_an_overlong_card_title_for_model_failover():
    try:
        ProfileAgent._normalize_card_title("这是一个非常冗长且不规范的完整句子")
    except ValueError:
        pass
    else:
        raise AssertionError("过长标题应触发模型切换")


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
        assert kwargs["tier"] == "fast"
        assert "不重复此前 assistant 已经给出的引导" in system_prompt
        assert "先用一个短句回应用户刚刚说清的具体行动或结果" in system_prompt
        assert "不像审核表或访谈提纲" in system_prompt
        assert "严禁虚构数字、成果、身份、职责" in system_prompt
        assert "suggested_replies" in system_prompt
        assert "校园项目" in user_prompt
        assert "负责访谈" in user_prompt
        return {
            "reply": "补充你如何从访谈记录中确定优先解决的问题，以及放弃了哪些备选方向。",
            "focus_dimension": "decision",
            "evidence_found": ["用户明确负责访谈"],
            "evidence_gap": "仍缺少方案取舍的判断依据。",
            "potential_hypotheses": ["可能具备基于证据进行产品取舍的潜能，仍需验证。"],
            "suggested_replies": ["我可以补充我当时筛选反馈时采用的标准。"],
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
    assert response.suggested_replies == ["我可以补充我当时筛选反馈时采用的标准。"]
    assert response.ready_for_proposal is False


def test_profile_agent_forwards_user_selected_model_tier():
    class TierGateway(ExplorationGateway):
        def generate_json(self, system_prompt: str, user_prompt: str, **kwargs) -> dict:
            assert kwargs["tier"] == "reasoning"
            return {
                "reply": "补充你整理需求时采用的筛选标准，以及这些标准如何影响后续方案。",
                "focus_dimension": "decision",
                "evidence_found": ["用户明确负责整理需求"],
                "evidence_gap": "仍缺少需求筛选的判断依据。",
                "potential_hypotheses": ["可能具备需求判断潜能，仍需验证。"],
                "ready_for_proposal": False,
            }

    asyncio.run(ProfileAgent(TierGateway()).explore(
        ProfileExplorationRequest(experience_text="我负责整理需求。", model_tier="reasoning"),
        "trace-tier",
    ))


class QuestionExplorationGateway(ExplorationGateway):
    def generate_json(self, system_prompt: str, user_prompt: str, **kwargs) -> dict:
        payload = super().generate_json(system_prompt, user_prompt, **kwargs)
        payload["reply"] = "你具体做了什么？"
        payload["focus_dimension"] = "unknown"
        return payload


def test_profile_agent_exploration_preserves_valid_question_reply():
    response = asyncio.run(ProfileAgent(QuestionExplorationGateway()).explore(
        ProfileExplorationRequest(
            experience_text="我在校园项目中负责访谈用户并整理需求，随后和团队完成了产品原型。",
        ),
        "trace-explore-invalid",
    ))

    assert response.focus_dimension == "ownership"
    assert response.reply == "你具体做了什么？"


def test_profile_agent_exploration_preserves_provider_reasoning_metadata():
    response = ProfileAgent._normalize_exploration(
        {
            "reply": "补充你当时的判断依据。",
            "focus_dimension": "decision",
            "evidence_gap": "仍缺少判断依据。",
            "_selected_model": "qwen3-30b-a3b-thinking-2507",
            "_thinking_enabled": True,
            "_reasoning_content": "先核对用户明确说出的行动，再决定追问维度。",
            "_reasoning_tokens": 17,
        },
        "trace-thinking",
    )

    assert response.thinking_enabled is True
    assert response.thinking_model == "qwen3-30b-a3b-thinking-2507"
    assert response.reasoning_content.startswith("先核对")
    assert response.reasoning_tokens == 17
    assert response.reasoning_status == "complete"
