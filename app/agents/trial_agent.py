import asyncio
import json
from typing import Any

from app.schemas.trial import (
    A02Answer,
    A02Task,
    TrialDimensionEvaluation,
    TrialEvaluation,
)
from app.schemas.task_catalog import DynamicTrialAnswer, TrialTaskDefinition
from app.services.llm_gateway import DashScopeQwenGateway


class TrialAgent:
    """Evaluate a completed fixed task against its source rubric."""

    SYSTEM_PROMPT = """你是“选择之前”的任务复盘助手。只评价用户在这一次固定任务中实际写下和做出的内容。
任务数据和评分维度来自任务库，不能新增任务、补写材料或编造企业真实数据。
不要按“标准答案”判对错；关注现象与原因是否分开、证据是否可追溯、优先级是否有影响面依据、验证动作是否能区分假设，以及事件后是否完成取舍。
按任务给定的每个隐藏Rubric输出0–100分项任务分，但不得计算总分，也不得把分项分直接当作用户能力等级。
主测能力必须对照任务给定的L1–L5行为锚点输出observed_level；证据不足时输出“证据不足”。辅测能力最多记录2项，而且只记录答案中明显出现的行为。
Coach提示不直接扣分；根据提示使用级别把coach_dependency标为独立完成、轻度提示、方向性提示或强提示，并主要反映在confidence。

面向用户的文字要求：
- summary 先肯定做得清楚的地方，再指出最重要的一个改进点，控制在 2–3 句话。
- evidence、level_reason、strengths、gaps 和 next_step 使用具体行为和原答案内容，不写空泛评价。
- next_step 必须是下一次可以直接执行的小动作。
- 避免“赋能、闭环、抓手、方法论、范式、拉通、颗粒度”等套话；必要的任务术语用一句普通话解释。
- 不把分数写成对人的总体评价，不使用“你就是/你不适合”之类结论。

只输出 JSON 对象，字段必须为：
{
  "summary": "",
  "dimensions": [{"dimension": "", "weight": 0, "score": 0, "evidence": ""}],
  "primary_ability": "",
  "observed_level": "L1|L2|L3|L4|L5|证据不足",
  "level_reason": "",
  "supporting_evidence": [{"ability": "", "observed_level": "L1|L2|L3|L4|L5|证据不足", "evidence": ""}],
  "process_evidence": [""],
  "coach_dependency": "独立完成|轻度提示|方向性提示|强提示",
  "strengths": [""],
  "gaps": [""],
  "next_step": "",
  "confidence": "低|中|高"
}
单次任务只形成 Observed Evidence，不得输出 Current Level、Potential Level、岗位胜任力认证、匹配百分比或真实企业结论。"""

    def __init__(self, gateway: DashScopeQwenGateway):
        self.gateway = gateway

    async def evaluate(self, task: A02Task, answer: A02Answer) -> TrialEvaluation:
        user_prompt = json.dumps(
            {
                "task_id": task.id,
                "task_title": task.title,
                "rubric": [item.model_dump(mode="json") for item in task.rubric],
                "answer": answer.model_dump(mode="json"),
            },
            ensure_ascii=False,
        )
        raw: dict[str, Any] = await asyncio.to_thread(
            self.gateway.generate_json,
            self.SYSTEM_PROMPT,
            user_prompt,
        )
        return TrialEvaluation.model_validate(raw)

    @staticmethod
    def _normalize_dynamic(
        task: TrialTaskDefinition,
        answer: DynamicTrialAnswer,
        raw: dict[str, Any],
    ) -> TrialEvaluation:
        evaluation = TrialEvaluation.model_validate(raw)
        rubric_by_dimension = {item.dimension: item for item in task.rubric}
        returned = {item.dimension: item for item in evaluation.dimensions}
        dimensions = [
            TrialDimensionEvaluation(
                dimension=criterion.dimension,
                weight=criterion.weight,
                score=returned.get(
                    criterion.dimension,
                    TrialDimensionEvaluation(
                        dimension=criterion.dimension,
                        score=0,
                        evidence="模型未返回该维度的有效证据。",
                    ),
                ).score,
                evidence=returned.get(
                    criterion.dimension,
                    TrialDimensionEvaluation(
                        dimension=criterion.dimension,
                        score=0,
                        evidence="模型未返回该维度的有效证据。",
                    ),
                ).evidence,
            )
            for criterion in rubric_by_dimension.values()
        ]
        allowed_supporting = set(task.supporting_skills[:2])
        supporting_evidence = [
            item for item in evaluation.supporting_evidence if item.ability in allowed_supporting
        ][:2]
        coach_level = max((item.level for item in answer.coach_usage), default=0)
        dependency = {
            0: "独立完成",
            1: "轻度提示",
            2: "方向性提示",
            3: "强提示",
        }[coach_level]
        confidence = evaluation.confidence
        if coach_level >= 2 and confidence == "高":
            confidence = "中"
        return evaluation.model_copy(
            update={
                "dimensions": dimensions,
                "primary_ability": task.primary_skill,
                "supporting_evidence": supporting_evidence,
                "coach_dependency": dependency,
                "confidence": confidence,
            }
        )

    async def evaluate_dynamic(
        self,
        task: TrialTaskDefinition,
        answer: DynamicTrialAnswer,
    ) -> TrialEvaluation:
        user_prompt = json.dumps(
            {
                "task_id": task.id,
                "task_title": task.title,
                "fixed_steps": [item.model_dump(mode="json") for item in task.steps],
                "event": task.event.model_dump(mode="json"),
                "rubric": [item.model_dump(mode="json") for item in task.rubric],
                "primary_ability": task.primary_skill,
                "supporting_abilities": task.supporting_skills[:2],
                "level_anchors": task.level_anchors,
                "answer": answer.model_dump(mode="json"),
            },
            ensure_ascii=False,
        )
        raw: dict[str, Any] = await asyncio.to_thread(
            self.gateway.generate_json,
            self.SYSTEM_PROMPT,
            user_prompt,
        )
        return self._normalize_dynamic(task, answer, raw)
