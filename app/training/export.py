"""Convert validated teacher labels into local SFT/DPO candidate files.

This module never invents a rejected answer. DPO pairs must be supplied by a
reviewer (or by a separate comparison process) so that a rejected response is
not silently fabricated from a teacher output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from app.evaluation.dataset import SftRecord
from app.schemas.trial import TrialEvaluation
from app.training.cases import (
    TrialCaseInput,
    build_teacher_system_prompt,
    build_teacher_user_prompt,
    normalize_case_payload,
)
from app.training.teacher import response_fingerprint


AUTO_SFT_STATUSES = frozenset({"silver_auto", "human_approved", "gold", "approved"})


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise ValueError(f"输入文件不存在：{source}")
    rows: list[dict[str, Any]] = []
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
        row = dict(payload)
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"无法读取 {source} 第 {line_number} 行：缺少 case_id")
        if case_id in seen_ids:
            raise ValueError(f"输入案例 case_id 重复：{case_id}")
        seen_ids.add(case_id)
        rows.append(row)
    if not rows:
        raise ValueError(f"输入文件为空：{source}")
    return rows


def _status(row: Mapping[str, Any]) -> str:
    validation = row.get("validation")
    if isinstance(validation, Mapping) and isinstance(validation.get("status"), str):
        return str(validation["status"])
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping) and isinstance(metadata.get("label_status"), str):
        return str(metadata["label_status"])
    return "unknown"


def _evaluation_payload(row: Mapping[str, Any]) -> dict[str, Any] | None:
    evaluation = row.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return None
    try:
        return TrialEvaluation.model_validate(evaluation).model_dump(mode="json")
    except Exception:
        return None


def _case_from_row(row: Mapping[str, Any], index: int) -> TrialCaseInput:
    return normalize_case_payload(row, index)


def _metadata(row: Mapping[str, Any], *, status: str) -> dict[str, str]:
    metadata: dict[str, str] = {
        "source": "teacher_label_export",
        "label_status": status,
    }
    teacher = row.get("teacher")
    if isinstance(teacher, Mapping):
        for key in ("model", "prompt_version", "request_fingerprint"):
            value = teacher.get(key)
            if value is not None:
                metadata[f"teacher_{key}"] = str(value)
    source_metadata = row.get("metadata")
    if isinstance(source_metadata, Mapping):
        if source_metadata.get("human_reviewed") is True:
            metadata["human_reviewed"] = "true"
        if isinstance(source_metadata.get("review_model_used"), bool):
            metadata["review_model_used"] = str(source_metadata["review_model_used"]).lower()
    return metadata


def to_sft_record(
    row: Mapping[str, Any],
    *,
    index: int,
    prompt_version: str,
    include_needs_review: bool = False,
) -> SftRecord | None:
    status = _status(row)
    source_metadata = row.get("metadata")
    human_reviewed = isinstance(source_metadata, Mapping) and source_metadata.get("human_reviewed") is True
    if status not in AUTO_SFT_STATUSES and not (
        include_needs_review and status == "needs_review" and human_reviewed
    ):
        return None
    evaluation = _evaluation_payload(row)
    if evaluation is None:
        return None
    case = _case_from_row(row, index)
    return SftRecord(
        case_id=case.case_id,
        task_id=case.task_id,
        messages=[
            {"role": "system", "content": build_teacher_system_prompt()},
            {
                "role": "user",
                "content": build_teacher_user_prompt(case, prompt_version=prompt_version),
            },
            {
                "role": "assistant",
                "content": json.dumps(evaluation, ensure_ascii=False, sort_keys=True),
            },
        ],
        metadata=_metadata(row, status=status),
    )


def export_sft_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    prompt_version: str,
    include_needs_review: bool = False,
) -> tuple[list[SftRecord], dict[str, int]]:
    records: list[SftRecord] = []
    counts = {"accepted": 0, "skipped": 0, "invalid": 0}
    for index, row in enumerate(rows, 1):
        try:
            record = to_sft_record(
                row,
                index=index,
                prompt_version=prompt_version,
                include_needs_review=include_needs_review,
            )
        except (TypeError, ValueError):
            counts["invalid"] += 1
            continue
        if record is None:
            counts["skipped"] += 1
            continue
        records.append(record)
        counts["accepted"] += 1
    return records, counts


def _dpo_evaluation(row: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    value = row.get(key)
    if isinstance(value, Mapping) and isinstance(value.get("evaluation"), Mapping):
        value = value["evaluation"]
    if not isinstance(value, Mapping):
        return None
    try:
        return TrialEvaluation.model_validate(value).model_dump(mode="json")
    except Exception:
        return None


def to_dpo_record(
    row: Mapping[str, Any],
    *,
    index: int,
    prompt_version: str,
    allow_raw_rejected: bool = False,
) -> dict[str, Any] | None:
    """Build an internal DPO ChatML record from an explicitly reviewed pair."""

    case = _case_from_row(row, index)
    chosen = _dpo_evaluation(row, "chosen_evaluation") or _dpo_evaluation(row, "chosen")
    rejected = _dpo_evaluation(row, "rejected_evaluation") or _dpo_evaluation(row, "rejected")
    raw_rejected: Mapping[str, Any] | None = None
    if rejected is None and allow_raw_rejected:
        candidate = row.get("rejected_evaluation") or row.get("rejected")
        if isinstance(candidate, Mapping):
            raw_rejected = candidate
            rejected = dict(candidate)
    if chosen is None or rejected is None:
        return None
    if response_fingerprint(chosen) == response_fingerprint(rejected):
        return None
    return {
        "case_id": case.case_id,
        "task_id": case.task_id,
        "messages": [
            {"role": "system", "content": build_teacher_system_prompt()},
            {
                "role": "user",
                "content": build_teacher_user_prompt(case, prompt_version=prompt_version),
            },
        ],
        "chosen": json.dumps(chosen, ensure_ascii=False, sort_keys=True),
        "rejected": json.dumps(rejected, ensure_ascii=False, sort_keys=True),
        "metadata": {
            "source": "human_reviewed_dpo_pair",
            "format": "dpo-chatml-v1",
            "rejected_schema_valid": raw_rejected is None,
        },
    }


def _assistant_message(value: Any) -> dict[str, str] | None:
    """Normalize an internal DPO output to Bailian's assistant message shape."""

    if isinstance(value, Mapping):
        if value.get("role") not in (None, "assistant"):
            return None
        content = value.get("content")
    else:
        content = value
    if not isinstance(content, str) or not content.strip():
        return None
    return {"role": "assistant", "content": content}


def to_bailian_dpo_record(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Project an internal DPO row into the strict Bailian upload format.

    ``case_id``, ``task_id`` and ``metadata`` are intentionally kept in the
    internal review artifact but are not accepted by Bailian's DPO importer.
    The importer also expects ``chosen`` and ``rejected`` to be assistant
    message objects rather than serialized JSON strings.
    """

    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    normalized_messages: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            return None
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            return None
        if not isinstance(content, str) or not content.strip():
            return None
        normalized_messages.append({"role": str(role), "content": content})
    if normalized_messages[-1]["role"] != "user":
        return None

    chosen = _assistant_message(record.get("chosen"))
    rejected = _assistant_message(record.get("rejected"))
    if chosen is None or rejected is None:
        return None
    return {
        "messages": normalized_messages,
        "chosen": chosen,
        "rejected": rejected,
    }


def export_dpo_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    prompt_version: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    counts = {"accepted": 0, "skipped": 0, "invalid": 0}
    for index, row in enumerate(rows, 1):
        try:
            record = to_dpo_record(row, index=index, prompt_version=prompt_version)
        except (TypeError, ValueError):
            counts["invalid"] += 1
            continue
        if record is None:
            counts["skipped"] += 1
            continue
        records.append(record)
        counts["accepted"] += 1
    return records, counts


def export_bailian_dpo_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    prompt_version: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Export DPO rows ready for direct upload to Bailian.

    The internal exporter remains available for review/debugging artifacts;
    this projection removes internal bookkeeping fields before upload.
    """

    internal_records, counts = export_dpo_records(rows, prompt_version=prompt_version)
    upload_records: list[dict[str, Any]] = []
    for internal_record in internal_records:
        record = to_bailian_dpo_record(internal_record)
        if record is None:
            counts["accepted"] -= 1
            counts["invalid"] += 1
            continue
        upload_records.append(record)
    return upload_records, counts


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    return destination
