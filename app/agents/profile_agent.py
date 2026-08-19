import asyncio
from typing import Any

from app.schemas.profile import (
    CardProposal,
    ExperienceSummary,
    ProfileProposalRequest,
    ProfileProposalResponse,
)
from app.services.llm_gateway import DashScopeQwenGateway


class ProfileAgent:
    """Generate candidate evidence cards; it never confirms or persists them."""

    SYSTEM_PROMPT = """你是“选择之前”的 ProfileAgent，只负责从用户授权提供的经历文字中提取候选证据。
你不能把候选内容当作已确认能力，也不能补写用户没有说过的事实。
严格只输出 JSON 对象，不要 Markdown，不要解释 JSON 以外的内容。
输出格式：
{
  "experience": {"title": "", "actions": [], "result": "", "source_refs": ["input:experience_text"]},
  "card_proposals": [
    {
      "title": "", "category": "洞察分析|产品策略|技术落地|数据驱动|协作沟通|交互体验",
      "description": "", "detail": "", "claim_level": "fact|interpretation|hypothesis",
      "evidence_type": "documented_fact|self_report|inference", "evidence_quote": "",
      "source_refs": ["input:experience_text"], "pending_verification": true,
      "next_verification": "", "match_reason": "", "workplace_application": ""
    }
  ],
  "next_question": "用陈述式补充提示，不使用问句"
}
最多输出 5 张卡。每张卡只表达一个主张。证据不足时降低为 hypothesis，并明确下一步验证。"""

    def __init__(self, gateway: DashScopeQwenGateway):
        self.gateway = gateway

    async def propose(
        self, request: ProfileProposalRequest, trace_id: str
    ) -> ProfileProposalResponse:
        user_prompt = self._build_prompt(request)
        raw = await asyncio.to_thread(
            self.gateway.generate_json, self.SYSTEM_PROMPT, user_prompt
        )
        return self._normalize(raw, trace_id)

    @staticmethod
    def _build_prompt(request: ProfileProposalRequest) -> str:
        target_role = request.target_role or "未指定目标岗位"
        existing = "、".join(request.existing_card_titles) or "暂无已确认卡牌"
        return (
            f"目标岗位：{target_role}\n"
            f"已有确认卡牌（仅用于避免重复，不代表本次证据）：{existing}\n"
            "以下是用户授权的经历文字，请只依据这段文字分析：\n"
            "--- BEGIN EXPERIENCE ---\n"
            f"{request.experience_text}\n"
            "--- END EXPERIENCE ---"
        )

    @staticmethod
    def _normalize(raw: dict[str, Any], trace_id: str) -> ProfileProposalResponse:
        experience_raw = raw.get("experience") or {}
        experience = ExperienceSummary(
            title=str(experience_raw.get("title") or "未命名经历")[:120],
            actions=[str(item)[:120] for item in (experience_raw.get("actions") or [])[:8]],
            result=(str(experience_raw["result"])[:500] if experience_raw.get("result") else None),
            source_refs=[str(item)[:120] for item in (experience_raw.get("source_refs") or [])[:10]]
            or ["input:experience_text"],
        )

        categories = {"洞察分析", "产品策略", "技术落地", "数据驱动", "协作沟通", "交互体验"}
        category_defaults = ["洞察分析", "产品策略", "协作沟通", "技术落地", "数据驱动"]
        color_by_category = {
            "洞察分析": ("purple", "Eye"),
            "产品策略": ("blue", "Layers"),
            "技术落地": ("emerald", "Sparkles"),
            "数据驱动": ("amber", "BarChart3"),
            "协作沟通": ("rose", "Users"),
            "交互体验": ("blue", "PanelsTopLeft"),
        }
        allowed_claim_levels = {"fact", "interpretation", "hypothesis"}
        allowed_evidence_types = {"documented_fact", "self_report", "inference"}
        cards: list[CardProposal] = []
        for index, item in enumerate((raw.get("card_proposals") or [])[:5]):
            if not isinstance(item, dict):
                continue
            category = str(item.get("category") or category_defaults[index % len(category_defaults)])
            if category not in categories:
                category = category_defaults[index % len(category_defaults)]
            color_tone, default_icon = color_by_category[category]
            evidence_quote = str(item.get("evidence_quote") or "用户自述，待进一步核验")[:500]
            title = str(item.get("title") or f"待确认能力 {index + 1}")[:80]
            claim_level = str(item.get("claim_level") or "interpretation")
            if claim_level not in allowed_claim_levels:
                claim_level = "interpretation"
            evidence_type = str(item.get("evidence_type") or "self_report")
            if evidence_type not in allowed_evidence_types:
                evidence_type = "self_report"
            cards.append(
                CardProposal(
                    id=f"proposal-{trace_id[:8]}-{index + 1}",
                    title=title,
                    category=category,  # type: ignore[arg-type]
                    description=str(item.get("description") or "从经历中提取的候选能力主张")[:240],
                    detail=str(item.get("detail") or evidence_quote)[:600],
                    icon=str(item.get("icon") or default_icon),
                    color_tone=color_tone,  # type: ignore[arg-type]
                    claim_level=claim_level,  # type: ignore[arg-type]
                    evidence_type=evidence_type,  # type: ignore[arg-type]
                    evidence_quote=evidence_quote,
                    source_refs=[str(ref)[:120] for ref in (item.get("source_refs") or [])[:10]]
                    or ["input:experience_text"],
                    pending_verification=bool(item.get("pending_verification", True)),
                    next_verification=str(item.get("next_verification") or "补充一个可观察的结果或下一步短任务")[:240],
                    match_reason=str(item.get("match_reason") or f"依据：{evidence_quote}")[:300],
                    workplace_application=str(item.get("workplace_application") or "需要在目标岗位任务中进一步验证")[:300],
                )
            )

        if not cards:
            raise ValueError("Qwen 未返回有效候选卡")
        follow_up = str(raw.get("next_question") or "").strip()
        if not follow_up or "？" in follow_up or "?" in follow_up:
            follow_up = "补充本人在该经历中具体负责的环节。"
        return ProfileProposalResponse(
            trace_id=trace_id,
            experience=experience,
            card_proposals=cards,
            next_question=follow_up[:300],
        )
