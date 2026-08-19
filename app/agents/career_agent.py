import asyncio
import json
from typing import Any

from app.knowledge.retriever import KnowledgeChunk
from app.schemas.career import (
    CareerCitation,
    CareerRecommendation,
    CareerSupport,
)
from app.schemas.profile import ProfileCard
from app.schemas.task_catalog import TrialTaskDefinition
from app.services.llm_gateway import DashScopeQwenGateway


class CareerAgent:
    """Explain one source-backed career path from confirmed profile cards."""

    SYSTEM_PROMPT = """你是“选择之前”的 CareerAgent，只负责解释一个职业探索路径。
只能依据用户已经确认的能力卡和给定的岗位知识片段回答，不能补写岗位事实、薪资、公司信息或录用结论。
本次目标岗位固定为“AI 产品经理”。下一步任务已经由后端的确定性选择器从固定任务库选出；你只能解释这个选择，不能改写任务或推荐其他任务。
每条支持性判断都要引用给定的 citation_id；不能引用不存在的 ID。
如果材料不足，必须写入 unknowns，不能用常识补齐。
不要输出匹配百分比、等级或“适合/不适合”的绝对结论。
只输出 JSON 对象，字段必须为：
{
  "summary": "",
  "supported": [{"claim": "", "card_ids": [], "citation_ids": []}],
  "unknowns": [],
  "confidence": "低|中|高"
}"""

    def __init__(self, gateway: DashScopeQwenGateway):
        self.gateway = gateway

    async def recommend(
        self,
        cards: list[ProfileCard],
        retrieved: list[KnowledgeChunk],
        next_task: TrialTaskDefinition,
        next_task_reason: str,
    ) -> CareerRecommendation:
        if not cards:
            raise ValueError("至少需要一张已确认能力卡。")
        if not retrieved:
            raise ValueError("岗位知识库没有返回可引用片段。")

        card_ids = {card.id for card in cards}
        citation_ids = {chunk.id for chunk in retrieved}
        raw: dict[str, Any] = await asyncio.to_thread(
            self.gateway.generate_json,
            self.SYSTEM_PROMPT,
            json.dumps(
                {
                    "target_role": "AI 产品经理",
                    "next_task": {
                        "id": next_task.id,
                        "title": next_task.title,
                        "primary_skill": next_task.primary_skill,
                        "selection_reason": next_task_reason,
                    },
                    "confirmed_cards": [
                        {
                            "id": card.id,
                            "title": card.title,
                            "category": card.category,
                            "description": card.description,
                            "detail": card.detail,
                            "evidence_quote": card.evidence_quote,
                            "claim_level": card.claim_level,
                            "evidence_type": card.evidence_type,
                        }
                        for card in cards
                    ],
                    "retrieved_context": [
                        {
                            "citation_id": chunk.id,
                            "document_title": chunk.document_title,
                            "source_locator": chunk.source_locator,
                            "trust_level": chunk.trust_level,
                            "content": chunk.content,
                        }
                        for chunk in retrieved
                    ],
                },
                ensure_ascii=False,
            ),
        )
        return self._normalize(
            raw,
            cards,
            retrieved,
            card_ids,
            citation_ids,
            next_task,
            next_task_reason,
        )

    @staticmethod
    def _normalize(
        raw: dict[str, Any],
        cards: list[ProfileCard],
        retrieved: list[KnowledgeChunk],
        card_ids: set[str],
        citation_ids: set[str],
        next_task: TrialTaskDefinition,
        next_task_reason: str,
    ) -> CareerRecommendation:
        supports: list[CareerSupport] = []
        for item in (raw.get("supported") or [])[:6]:
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim") or "").strip()[:300]
            if not claim:
                continue
            supports.append(
                CareerSupport(
                    claim=claim,
                    card_ids=[
                        str(card_id)
                        for card_id in (item.get("card_ids") or [])
                        if str(card_id) in card_ids
                    ][:4],
                    citation_ids=[
                        str(citation_id)
                        for citation_id in (item.get("citation_ids") or [])
                        if str(citation_id) in citation_ids
                    ][:5],
                )
            )

        if not supports:
            supports.append(
                CareerSupport(
                    claim=f"已确认的“{cards[0].title}”可作为 AI 产品经理方向的待验证基础。",
                    card_ids=[cards[0].id],
                    citation_ids=[retrieved[0].id],
                )
            )

        confidence = str(raw.get("confidence") or "中")
        if confidence not in {"低", "中", "高"}:
            confidence = "中"
        summary = str(
            raw.get("summary")
            or f"当前材料支持进入 AI 产品经理试路，下一步通过 {next_task.id} 验证未知项。"
        )[:500]
        unknowns = [
            str(item).strip()[:240]
            for item in (raw.get("unknowns") or [])[:6]
            if str(item).strip()
        ]
        return CareerRecommendation(
            summary=summary,
            supported=supports,
            unknowns=unknowns or [f"尚未通过 {next_task.id} 验证{next_task.primary_skill}。"],
            next_task_id=next_task.id,
            next_task_title=next_task.title,
            next_task_reason=next_task_reason,
            confidence=confidence,  # type: ignore[arg-type]
            citations=[
                CareerCitation(
                    id=chunk.id,
                    document_title=chunk.document_title,
                    source_locator=chunk.source_locator,
                    content=chunk.content,
                    trust_level=chunk.trust_level,
                    source_note=chunk.source_note,
                )
                for chunk in retrieved
            ],
        )
