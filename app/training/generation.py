"""Generate candidate text-only TrialAgent answers with a cached Qwen call.

Generated answers are silver input candidates, not gold labels. They must pass
schema validation and later go through the teacher-evaluation and human-audit
pipeline before they can enter SFT/DPO data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from app.config import Settings, get_settings
from app.services.llm_gateway import DashScopeQwenGateway
from app.tasks.catalog import get_task_definition
from app.training.teacher import TeacherCache


GENERATION_SYSTEM_PROMPT = """你是固定任务库的作答案例构造器，只生成文本型 DynamicTrialAnswer。
任务定义、材料、五步 Schema、事件和 Rubric 都是不可修改的输入。不要新增任务、修改题目、
编造真实企业数据、生成评价或能力等级。根据请求的质量级别填写用户在真实任务中可能写下的
作答；每个字段都应是可核对的具体行动、判断或交付物。必须输出 step_answers，至少填写一个
给定步骤；不能只返回空对象、评价字段或自定义字段。只输出一个合法 JSON 对象。"""


@dataclass(frozen=True)
class GeneratedCaseResponse:
    raw: dict[str, Any] | None
    task_id: str
    quality_level: str
    model: str
    prompt_version: str
    fingerprint: str
    cache_hit: bool
    api_calls: int
    status: str = "ok"


def build_case_generation_prompt(
    task_id: str,
    *,
    quality_level: str,
    prompt_version: str,
) -> str:
    task = get_task_definition(task_id)
    payload = {
        "prompt_version": prompt_version,
        "quality_level": quality_level,
        "quality_level_definition": {
            "L1": "只写零散意图，缺少可执行步骤和证据",
            "L2": "完成部分步骤，但证据、取舍或事件响应不完整",
            "L3": "完成主要步骤，能引用材料并给出基本验证动作",
            "L4": "步骤、证据、优先级和事件调整均较完整",
            "L5": "形成可复核的完整交付物，明确边界、对照和风险取舍",
        }[quality_level],
        "instruction": "只生成答案，不生成评价；答案必须严格使用给定五步 ID 和事件字段。",
        "required_output_shape": {
            "selected_card_ids": "string[]，没有能力卡时为空数组",
            "card_play_rounds": "object[]，没有出牌记录时为空数组",
            "step_answers": {step.id: "string，填写该步骤的作答" for step in task.steps},
            "event_decision": "维持 或 调整",
            "event_response": "string，说明事件后的取舍和依据",
        },
        "task": {
            "task_id": task.id,
            "title": task.title,
            "role": task.role,
            "background": task.background,
            "goal": task.goal,
            "constraints": task.constraints,
            "materials": [item.model_dump(mode="json") for item in task.materials],
            "steps": [item.model_dump(mode="json") for item in task.steps],
            "event": task.event.model_dump(mode="json"),
            "level_anchors": task.level_anchors,
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def generation_fingerprint(
    task_id: str,
    *,
    quality_level: str,
    model: str,
    prompt_version: str,
) -> str:
    body = {
        "task_id": task_id,
        "quality_level": quality_level,
        "model": model,
        "prompt_version": prompt_version,
    }
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class CaseGenerator:
    """Generate candidate answers with one idempotent text-model request."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        cache: TeacherCache | None = None,
        gateway: DashScopeQwenGateway | None = None,
    ):
        self.settings = settings or get_settings()
        self.cache = cache or TeacherCache(self.settings.trial_teacher_cache_path)
        self.gateway = gateway or DashScopeQwenGateway(self.settings)

    def generate(
        self,
        task_id: str,
        *,
        quality_level: str,
        model: str | None = None,
        prompt_version: str | None = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> GeneratedCaseResponse:
        if quality_level not in {"L1", "L2", "L3", "L4", "L5"}:
            raise ValueError("quality_level 必须是 L1-L5")
        task = get_task_definition(task_id)
        selected_model = (model or self.settings.trial_teacher_model).strip()
        selected_prompt_version = (
            prompt_version or self.settings.trial_teacher_prompt_version
        ).strip()
        if not selected_model:
            raise ValueError("案例生成模型不能为空")
        fingerprint = generation_fingerprint(
            task.id,
            quality_level=quality_level,
            model=selected_model,
            prompt_version=selected_prompt_version,
        )
        with self.cache.lock(fingerprint):
            if not force:
                cached = self.cache.get(fingerprint)
                if cached is not None:
                    return GeneratedCaseResponse(
                        raw=cached,
                        task_id=task.id,
                        quality_level=quality_level,
                        model=selected_model,
                        prompt_version=selected_prompt_version,
                        fingerprint=fingerprint,
                        cache_hit=True,
                        api_calls=0,
                    )
            if dry_run:
                return GeneratedCaseResponse(
                    raw=None,
                    task_id=task.id,
                    quality_level=quality_level,
                    model=selected_model,
                    prompt_version=selected_prompt_version,
                    fingerprint=fingerprint,
                    cache_hit=False,
                    api_calls=0,
                    status="planned",
                )
            raw = self.gateway.generate_json(
                GENERATION_SYSTEM_PROMPT,
                build_case_generation_prompt(
                    task.id,
                    quality_level=quality_level,
                    prompt_version=selected_prompt_version,
                ),
                model=selected_model,
            )
            self.cache.put(
                fingerprint,
                model_id=selected_model,
                prompt_version=selected_prompt_version,
                response=raw,
            )
            return GeneratedCaseResponse(
                raw=raw,
                task_id=task.id,
                quality_level=quality_level,
                model=selected_model,
                prompt_version=selected_prompt_version,
                fingerprint=fingerprint,
                cache_hit=False,
                api_calls=1,
            )
