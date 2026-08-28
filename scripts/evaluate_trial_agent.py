"""Run or aggregate the four TrialAgent evaluation arms.

Offline aggregation is the default. Live Qwen calls require ``--live``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.evaluation.report import write_json, write_markdown  # noqa: E402
from app.evaluation.runner import (  # noqa: E402
    load_prediction_records,
    run_live_evaluation,
    run_offline_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 TrialAgent 四组对照评测报告")
    parser.add_argument("--cases", required=True, type=Path, help="评测集 JSONL")
    parser.add_argument("--predictions", type=Path, help="离线预测结果 JSONL；不传则必须使用 --live")
    parser.add_argument("--live", action="store_true", help="显式调用百炼 Qwen，按每个 arm/案例调用一次")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation-results/trial-agent-v1"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.live and args.predictions:
        print("--live 与 --predictions 不能同时使用。", file=sys.stderr)
        return 2
    if not args.live and not args.predictions:
        print("默认仅做离线汇总，请提供 --predictions；需要真实调用时显式使用 --live。", file=sys.stderr)
        return 2
    try:
        if args.live:
            report = asyncio.run(run_live_evaluation(args.cases))
        else:
            report = run_offline_evaluation(args.cases, load_prediction_records(args.predictions))
        json_path = write_json(report, args.output_dir / "report.json")
        markdown_path = write_markdown(report, args.output_dir / "report.md")
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"报告 JSON：{json_path}")
    print(f"报告 Markdown：{markdown_path}")
    print(f"对照方案：{'、'.join(summary.arm for summary in report.arms)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
