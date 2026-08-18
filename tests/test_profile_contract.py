from app.agents.profile_agent import ProfileAgent
from app.schemas.profile import ProfileProposalRequest


class FakeGateway:
    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
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


async def test_profile_agent_returns_pending_cards():
    response = await ProfileAgent(FakeGateway()).propose(
        ProfileProposalRequest(experience_text="我在校园项目中访谈用户并根据反馈调整了方案，最后完成了可用原型。"),
        "trace-test-1234",
    )
    assert response.card_proposals[0].title == "用户研究"
    assert response.card_proposals[0].pending_verification is True
    assert response.card_proposals[0].source_refs == ["input:experience_text"]
