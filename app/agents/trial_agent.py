import asyncio
import json
from typing import Any

from app.schemas.trial import A02Answer, A02Task, TrialEvaluation
from app.services.llm_gateway import DashScopeQwenGateway


class TrialAgent:
    """Evaluate a completed fixed task against its source rubric."""

    SYSTEM_PROMPT = """你是 CoachAgent 的任务评价模块，只评价用户在固定任务中的可观察行为。
任务数据和评分维度来自任务库，不能新增任务、补写材料或编造企业真实数据。
不要按“标准答案”判对错；关注现象与原因是否分开、证据是否可追溯、优先级是否有影响面依据、验证动作是否能区分假设，以及事件后是否完成取舍。
只输出 JSON 对象，字段必须为：
{
  "summary": "",
  "dimensions": [{"dimension": "", "score": 0, "evidence": ""}],
  "strengths": [""],
  "gaps": [""],
  "next_step": "",
  "confidence": "低|中|高"
}
单次任务只形成 Observed Evidence，不得输出 Current Level、岗位胜任力认证或真实企业结论。"""

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
