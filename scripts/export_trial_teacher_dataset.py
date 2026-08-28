"""Export validated teacher labels for local SFT and explicitly reviewed DPO pairs.

The script is offline. It never calls Qwen and never fabricates a rejected DPO
answer. Rows needing review are exported only when they carry
``metadata.human_reviewed=true`` and ``--include-needs-review`` is supplied.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.config import get_settings  # noqa: E402
from app.training.export import (  # noqa: E402
    export_dpo_records,
    export_sft_records,
    read_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 TrialAgent 教师标签的 SFT/DPO 候选数据")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="build_trial_teacher_labels.py 生成的教师标签 JSONL",
    )
    parser.add_argument(
        "--sft-output",
        type=Path,
        default=Path("datasets/trial_agent/v1/teacher_sft.local.jsonl"),
        help="SFT ChatML 候选输出；默认写入本地忽略路径",
    )
    parser.add_argument(
        "--dpo-input",
        type=Path,
        help="人工审核的 DPO 对比 JSONL；每行需提供 chosen_evaluation 和 rejected_evaluation",
    )
    parser.add_argument(
        "--dpo-output",
        type=Path,
        default=Path("datasets/trial_agent/v1/teacher_dpo.local.jsonl"),
        help="DPO 候选输出；默认写入本地忽略路径",
    )
    parser.add_argument(
        "--prompt-version",
        help="SFT/DPO 用户输入 Prompt 版本；默认读取 TRIAL_TEACHER_PROMPT_VERSION",
    )
    parser.add_argument(
        "--include-needs-review",
        action="store_true",
        help="仅在 metadata.human_reviewed=true 时纳入 needs_review 记录",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        labels = read_jsonl(args.input)
        dpo_rows = read_jsonl(args.dpo_input) if args.dpo_input else []
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    settings = get_settings()
    prompt_version = (args.prompt_version or settings.trial_teacher_prompt_version).strip()
    if not prompt_version:
        print("Prompt 版本不能为空。", file=sys.stderr)
        return 2

    sft_records, sft_counts = export_sft_records(
        labels,
        prompt_version=prompt_version,
        include_needs_review=args.include_needs_review,
    )
    write_jsonl(
        args.sft_output,
        ({
            "case_id": record.case_id,
            "task_id": record.task_id,
            "messages": [message.model_dump(mode="json") for message in record.messages],
            "metadata": record.metadata,
        } for record in sft_records),
    )
    print(
        f"SFT：接受 {sft_counts['accepted']} 条，跳过 {sft_counts['skipped']} 条，"
        f"无效 {sft_counts['invalid']} 条 -> {args.sft_output}"
    )

    if args.dpo_input:
        dpo_records, dpo_counts = export_dpo_records(
            dpo_rows,
            prompt_version=prompt_version,
        )
        write_jsonl(args.dpo_output, dpo_records)
        print(
            f"DPO：接受 {dpo_counts['accepted']} 条，跳过 {dpo_counts['skipped']} 条，"
            f"无效 {dpo_counts['invalid']} 条 -> {args.dpo_output}"
        )
    else:
        print("DPO：未提供人工审核 pair，未生成拒答样本。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
