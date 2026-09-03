"""Combine completed TrialAgent, RAG and multimodal reports without API calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.experiment_evidence import (  # noqa: E402
    build_experiment_evidence_report,
    render_experiment_evidence_markdown,
)
from app.evaluation.unified import load_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总核心技术实验证据（离线，不调用模型）")
    parser.add_argument("--trial-report", type=Path, required=True)
    parser.add_argument("--rag-report", type=Path, required=True)
    parser.add_argument("--multimodal-report", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation-results" / "evidence-v1")
    args = parser.parse_args()

    trial = load_json(args.trial_report)
    rag = load_json(args.rag_report)
    multimodal = [load_json(path) for path in args.multimodal_report]
    if trial is None or rag is None or any(report is None for report in multimodal):
        raise ValueError("所有输入报告都必须存在")
    report = build_experiment_evidence_report(
        trial=trial,
        rag=rag,
        multimodal=[item for item in multimodal if item is not None],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "report.json"
    markdown_path = args.output_dir / "report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_experiment_evidence_markdown(report), encoding="utf-8")
    print(f"已写入：{json_path}")
    print(f"已写入：{markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
