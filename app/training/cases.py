from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from pydantic import ValidationError

from app.schemas.task_catalog import DynamicTrialAnswer
from app.services.trial_scoring import TrialScoringService
from app.tasks.catalog import get_task_definition


@dataclass(frozen=True)
class TrialCaseInput:
    """A raw, text-only case ready for teacher evaluation."""

    case_id: str
    task_id: str
    answer: DynamicTrialAnswer
    confirmed_card_ids: tuple[str, ...] = ()
    evidence_catalog: tuple[dict[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def request_payload(self) -> dict[str, Any]:
        task = get_task_definition(self.task_id)
        return {
            "task_id": task.id,
            "task_title": task.title,
            "role": task.role,
            "background": task.background,
            "goal": task.goal,
            "constraints": task.constraints,
            "fixed_steps": [step.model_dump(mode="json") for step in task.steps],
            "event": task.event.model_dump(mode="json"),
            "rubric": [criterion.model_dump(mode="json") for criterion in task.rubric],
            "level_anchors": task.level_anchors,
            "confirmed_ability_cards": list(self.confirmed_card_ids),
            "evidence_catalog": list(self.evidence_catalog),
            "answer": self.answer.model_dump(mode="json"),
        }

    def as_record(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "task_id": self.task_id,
            "answer": self.answer.model_dump(mode="json"),
            "confirmed_card_ids": list(self.confirmed_card_ids),
            "evidence_catalog": list(self.evidence_catalog),
            "metadata": dict(self.metadata),
        }


def _json_message_content(message: Mapping[str, Any]) -> Any:
    content = message.get("content")
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, Mapping)
        )
    return content


def _parse_message_payload(messages: Any) -> dict[str, Any]:
    if not isinstance(messages, list):
        return {}
    for message in reversed(messages):
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = _json_message_content(message)
        if isinstance(content, Mapping):
            return dict(content)
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, Mapping):
                return dict(parsed)
    return {}


def _normalise_catalog(
    task_id: str,
    answer: DynamicTrialAnswer,
    supplied: Any,
) -> tuple[dict[str, Any], ...]:
    if isinstance(supplied, list):
        items: list[dict[str, Any]] = []
        for index, item in enumerate(supplied):
            if isinstance(item, Mapping):
                item_id = item.get("id")
                if isinstance(item_id, str) and item_id.strip():
                    items.append(dict(item))
            elif isinstance(item, str) and item.strip():
                items.append({"id": item.strip(), "content": item.strip()})
        if items:
            return tuple(items)

    task = get_task_definition(task_id)
    bundle = TrialScoringService.build_evidence(task, answer, [])
    return tuple(item.model_dump(mode="json") for item in bundle.items)


def normalize_case_payload(payload: Mapping[str, Any], index: int) -> TrialCaseInput:
    row = dict(payload)
    message_payload = _parse_message_payload(row.get("messages"))
    merged = {**message_payload, **row}
    task_payload = merged.get("task")
    task_id_value = merged.get("task_id")
    if not task_id_value and isinstance(task_payload, Mapping):
        task_id_value = task_payload.get("id")
    task_id = str(task_id_value or "").strip()
    if not task_id:
        raise ValueError(f"第 {index} 行缺少 task_id")
    task = get_task_definition(task_id)
    answer_payload = merged.get("answer") or merged.get("task_answer") or message_payload.get("answer") or {}
    try:
        answer = DynamicTrialAnswer.model_validate(answer_payload)
    except ValidationError as exc:
        raise ValueError(f"第 {index} 行的 answer 不符合 DynamicTrialAnswer：{exc}") from exc
    case_id = str(merged.get("case_id") or merged.get("id") or f"case-{index}").strip()
    if not case_id:
        raise ValueError(f"第 {index} 行的 case_id 不能为空")
    cards = merged.get("confirmed_card_ids") or merged.get("selected_card_ids") or []
    if not isinstance(cards, list):
        cards = []
    metadata = merged.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    metadata = {str(key): value for key, value in metadata.items()}
    metadata.setdefault("source", "case_input")
    catalog = _normalise_catalog(task.id, answer, merged.get("evidence_catalog"))
    return TrialCaseInput(
        case_id=case_id,
        task_id=task.id,
        answer=answer,
        confirmed_card_ids=tuple(str(card_id) for card_id in cards if str(card_id).strip()),
        evidence_catalog=catalog,
        metadata=metadata,
    )


def load_case_inputs(path: str | Path) -> list[TrialCaseInput]:
    source = Path(path)
    if not source.exists():
        raise ValueError(f"输入文件不存在：{source}")
    cases: list[TrialCaseInput] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"无法读取 {source} 第 {line_number} 行：{exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"无法读取 {source} 第 {line_number} 行：顶层必须是对象")
        case = normalize_case_payload(payload, line_number)
        if case.case_id in seen_ids:
            raise ValueError(f"输入案例 case_id 重复：{case.case_id}")
        seen_ids.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError(f"输入文件为空：{source}")
    return cases


def build_teacher_system_prompt() -> str:
    from app.agents.trial_agent import TrialAgent

    return TrialAgent.SYSTEM_PROMPT


def build_teacher_user_prompt(case: TrialCaseInput, *, prompt_version: str) -> str:
    payload = {
        "prompt_version": prompt_version,
        "instruction": "只评价当前案例，不生成新的任务或答案；只返回 TrialEvaluation JSON 对象。",
        **case.request_payload(),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def case_fingerprint(
    case: TrialCaseInput,
    *,
    model: str,
    prompt_version: str,
) -> str:
    body = {
        "model": model,
        "prompt_version": prompt_version,
        "request": case.request_payload(),
    }
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
