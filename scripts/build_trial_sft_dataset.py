"""Partition human-reviewed TrialAgent ChatML records for Bailian SFT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.evaluation.dataset import dataset_sha256, read_sft_jsonl, split_sft_records, write_sft_splits  # noqa: E402
from app.tasks.catalog import list_task_definitions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将人工审核的 TrialAgent 记录拆分为 Bailian SFT 数据集")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="人工审核 JSONL；每行包含 case_id、task_id、messages",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("datasets/trial_agent/v1/generated"),
        type=Path,
        help="生成目录；默认位于 datasets/trial_agent/v1/generated",
    )
    parser.add_argument(
        "--holdout-task",
        action="append",
        dest="holdout_tasks",
        help="锁定测试任务，可重复传入；未指定时默认留出任务库排序后的最后两个任务",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="非锁定任务中划入 validation 的比例，默认 0.2",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"输入文件不存在：{args.input}", file=sys.stderr)
        return 2
    records = read_sft_jsonl(args.input)
    default_holdout = {task.id for task in sorted(list_task_definitions(), key=lambda item: item.id)[-2:]}
    holdout_tasks = set(args.holdout_tasks or default_holdout)
    known_tasks = {task.id for task in list_task_definitions()}
    unknown = holdout_tasks - known_tasks
    if unknown:
        print(f"锁定测试任务不在固定任务库中：{'、'.join(sorted(unknown))}", file=sys.stderr)
        return 2
    splits = split_sft_records(
        records,
        holdout_task_ids=holdout_tasks,
        validation_ratio=args.validation_ratio,
    )
    paths = write_sft_splits(splits, args.output_dir)
    print(f"数据集版本：trial-agent-sft-v1")
    print(f"锁定任务：{'、'.join(sorted(holdout_tasks))}")
    for split_name in ("train", "validation", "test_locked"):
        print(f"{split_name}: {len(splits[split_name])} 条，{paths[split_name]}")
    print(f"输入数据 SHA-256：{dataset_sha256(args.input)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
