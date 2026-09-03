from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.tasks.catalog import get_task_definition


class SftMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=60_000)


class SftRecord(BaseModel):
    """A human-reviewed ChatML record accepted by Bailian SFT."""

    case_id: str = Field(min_length=1, max_length=120)
    task_id: str
    messages: list[SftMessage] = Field(min_length=2, max_length=32)
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_record(self) -> "SftRecord":
        try:
            get_task_definition(self.task_id)
        except KeyError as exc:
            raise ValueError(f"任务不在固定任务库中：{self.task_id}") from exc
        if self.messages[-1].role != "assistant":
            raise ValueError("SFT 记录最后一条消息必须是 assistant 标注")
        try:
            parsed = json.loads(self.messages[-1].content)
        except json.JSONDecodeError as exc:
            raise ValueError("assistant 标注必须是合法 JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("assistant 标注的 JSON 顶层必须是对象")
        return self


def read_sft_jsonl(path: str | Path) -> list[SftRecord]:
    records: list[SftRecord] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            record = SftRecord.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise ValueError(f"无法读取 {path} 第 {line_number} 行：{exc}") from exc
        records.append(record)
    if not records:
        raise ValueError(f"{path} 不包含可用标注记录")
    return records


def split_sft_records(
    records: Iterable[SftRecord],
    *,
    holdout_task_ids: set[str],
    validation_ratio: float = 0.2,
) -> dict[str, list[SftRecord]]:
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio 必须位于 [0, 1) 区间")
    unique_case_ids: set[str] = set()
    train_pool: list[SftRecord] = []
    locked: list[SftRecord] = []
    for record in records:
        if record.case_id in unique_case_ids:
            raise ValueError(f"case_id 重复：{record.case_id}")
        unique_case_ids.add(record.case_id)
        (locked if record.task_id in holdout_task_ids else train_pool).append(record)
    train: list[SftRecord] = []
    validation: list[SftRecord] = []
    for record in sorted(train_pool, key=lambda item: item.case_id):
        digest = hashlib.sha256(record.case_id.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") / 2**32
        (validation if bucket < validation_ratio else train).append(record)
    # Tiny annotation batches should still produce a validation row when the
    # caller requested validation and there is more than one non-holdout case.
    if validation_ratio > 0 and len(train_pool) > 1 and not validation:
        validation.append(train.pop())
    return {"train": train, "validation": validation, "test_locked": sorted(locked, key=lambda item: item.case_id)}


def write_sft_splits(
    splits: dict[str, list[SftRecord]],
    output_dir: str | Path,
) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split_name in ("train", "validation", "test_locked"):
        records = splits.get(split_name, [])
        path = directory / f"{split_name}.jsonl"
        lines = [
            json.dumps({"messages": [message.model_dump(mode="json") for message in record.messages]}, ensure_ascii=False)
            for record in records
        ]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        paths[split_name] = path

    manifest = {
        "dataset_version": "trial-agent-sft-v1",
        "format": "messages-jsonl",
        "holdout_task_ids": sorted({record.task_id for record in splits.get("test_locked", [])}),
        "counts": {name: len(splits.get(name, [])) for name in ("train", "validation", "test_locked")},
        "case_ids": {
            name: [record.case_id for record in splits.get(name, [])]
            for name in ("train", "validation", "test_locked")
        },
        "source_policy": "只接收固定12任务库的人工审核记录；脚本不生成训练标签。",
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths["manifest"] = manifest_path
    return paths


def dataset_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
