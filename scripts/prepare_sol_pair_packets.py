"""Prepare one review packet per baseline/teacher evaluation pair.

The packet is the hand-off boundary for one independent model review task.
It contains one case, the deterministic evidence catalog, and the two Qwen
evaluations that must be compared.  The reviewer writes a separate JSON file;
the packet itself is never modified in place.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.tasks.catalog import get_task_definition  # noqa: E402
from app.training.cases import TrialCaseInput, load_case_inputs  # noqa: E402
from app.training.export import read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为独立评价子任务准备逐案例对比包")
    parser.add_argument("--cases", required=True, type=Path, help="案例输入 JSONL")
    parser.add_argument("--teacher", required=True, type=Path, help="强化版 Qwen 评价 JSONL")
    parser.add_argument("--baseline", required=True, type=Path, help="基础版 Qwen 评价 JSONL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/trial_agent/v1/sol_review_packets.local"),
        help="逐案例 packet 输出目录",
    )
    parser.add_argument("--resume", action="store_true", help="跳过已有 packet")
    parser.add_argument("--limit", type=int, help="最多准备多少个 packet")
    parser.add_argument(
        "--review-model",
        default="gpt-5.6-luna",
        help="写入复核契约的模型，默认 gpt-5.6-luna",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="max",
        help="写入复核契约的推理强度，默认 max",
    )
    return parser.parse_args()


def _rows_by_case(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["case_id"]): row for row in read_jsonl(path)}


def _safe_name(case_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id)


def _packet(
    case: TrialCaseInput,
    baseline: dict[str, Any],
    teacher: dict[str, Any],
    *,
    review_model: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    task = get_task_definition(case.task_id)
    return {
        "packet_version": "pair-review-v2",
        "pair_id": f"eval-pair-{case.case_id}",
        "case_id": case.case_id,
        "task_id": case.task_id,
        "task": {
            "id": task.id,
            "title": task.title,
            "role": task.role,
            "background": task.background,
            "goal": task.goal,
            "constraints": task.constraints,
            "fixed_steps": [step.model_dump(mode="json") for step in task.steps],
            "event": task.event.model_dump(mode="json"),
            "rubric": [criterion.model_dump(mode="json") for criterion in task.rubric],
            "level_anchors": task.level_anchors,
        },
        "case": case.request_payload(),
        "baseline": {
            "evaluation": baseline.get("evaluation"),
            "validation": baseline.get("validation"),
            "teacher": baseline.get("teacher"),
        },
        "teacher": {
            "evaluation": teacher.get("evaluation"),
            "validation": teacher.get("validation"),
            "teacher": teacher.get("teacher"),
        },
        "review_contract": {
            "model": review_model,
            "reasoning_effort": reasoning_effort,
            "instruction": (
                "只评价这一个 baseline/teacher 对比对。先检查两份评价是否忠实引用 case 中的答案和证据目录，"
                "再输出一份结构合法的 enhanced_evaluation。明确选择 chosen_source 和 rejected_source；"
                "只有存在可解释的质量差异、证据边界清楚且两份输出不是重复内容时 pair_valid 才能为 true。"
                "不得新增证据 ID、能力卡、任务材料或 Rubric 维度；无法判定时 pair_valid=false。"
            ),
            "required_output": {
                "case_id": case.case_id,
                "pair_valid": False,
                "chosen_source": "teacher|baseline|enhanced|none",
                "rejected_source": "teacher|baseline|none",
                "enhanced_evaluation": "TrialEvaluation JSON；不能引用不存在的证据",
                "rationale": "说明质量差异、证据依据和排除原因",
                "issues": ["可选的问题代码"],
                "review_model": review_model,
                "reasoning_effort": reasoning_effort,
            },
        },
    }


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        print("--limit 必须大于 0", file=sys.stderr)
        return 2
    try:
        cases = load_case_inputs(args.cases)
        teacher_rows = _rows_by_case(args.teacher)
        baseline_rows = _rows_by_case(args.baseline)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    selected = cases if args.limit is None else cases[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    for case in selected:
        baseline = baseline_rows.get(case.case_id)
        teacher = teacher_rows.get(case.case_id)
        if baseline is None or teacher is None:
            skipped += 1
            print(f"跳过 {case.case_id}：缺少 baseline 或 teacher 评价", file=sys.stderr)
            continue
        destination = args.output_dir / f"{_safe_name(case.case_id)}.json"
        if args.resume and destination.exists():
            continue
        destination.write_text(
            json.dumps(
                _packet(
                    case,
                    baseline,
                    teacher,
                    review_model=args.review_model,
                    reasoning_effort=args.reasoning_effort,
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        written += 1
    print(f"packet：{written}，跳过：{skipped}，目录：{args.output_dir}")
    return 0 if skipped == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
