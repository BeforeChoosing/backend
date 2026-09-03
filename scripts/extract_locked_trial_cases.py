"""Extract a small, deterministic locked evaluation set from DPO pairs.

The extracted cases are evaluation inputs only. They are never used as DPO
training data and retain the chosen evaluation as a provisional reference
that must be spot-checked before publishing metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _parse_pair(line: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    row = json.loads(line)
    messages = row.get("messages") or []
    user = next(item for item in messages if item.get("role") == "user")
    payload = json.loads(user["content"])
    task_id = str(payload["task_id"])
    answer = payload["answer"]
    chosen = json.loads(row["chosen"]["content"])
    return task_id, answer, chosen


def _gold(chosen: dict[str, Any]) -> dict[str, Any]:
    dimensions = {
        str(item["dimension"]): int(item["score"])
        for item in chosen.get("dimensions", [])
        if isinstance(item, dict) and item.get("dimension") is not None and isinstance(item.get("score"), int)
    }
    refs: list[str] = []
    for item in chosen.get("dimensions", []):
        if isinstance(item, dict):
            refs.extend(str(ref) for ref in item.get("evidence_refs", []) if isinstance(ref, str))
    refs.extend(str(ref) for ref in chosen.get("evidence_refs", []) if isinstance(ref, str))
    ability_ids = [
        str(item["card_id"])
        for item in chosen.get("ability_applications", [])
        if isinstance(item, dict) and item.get("card_id")
    ]
    return {
        "dimensions": dimensions,
        "observed_level": chosen.get("observed_level", "证据不足"),
        "evidence_refs": sorted(set(refs)),
        "required_ability_applications": sorted(set(ability_ids)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="从 DPO 对中抽取锁定 TrialAgent 评测集")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=2)
    args = parser.parse_args()

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task_id, answer, chosen = _parse_pair(line)
        if len(grouped[task_id]) < args.per_task:
            grouped[task_id].append((answer, chosen))
    if len(grouped) != 12 or any(len(items) < args.per_task for items in grouped.values()):
        raise ValueError("DPO 数据未覆盖 12 个任务，无法生成完整锁定集")

    rows: list[dict[str, Any]] = []
    for task_id in sorted(grouped):
        for index, (answer, chosen) in enumerate(grouped[task_id], 1):
            rows.append({
                "case_id": f"dpo-{task_id.lower()}-{index:02d}",
                "task_id": task_id,
                "answer": answer,
                "gold": _gold(chosen),
                "confirmed_card_ids": list(answer.get("selected_card_ids", [])),
                "metadata": {
                    "source": "sol_dpo_pairs.local.jsonl",
                    "fixture_type": "locked_eval_extracted",
                    "label_status": "provisional_review_required",
                    "training": False,
                },
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"已抽取 {len(rows)} 条锁定评测案例：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
