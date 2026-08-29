"""Aggregate resumable TrialAgent batch reports without new API calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.trial_agent import TrialAgent  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.evaluation.dataset import dataset_sha256  # noqa: E402
from app.evaluation.models import CaseEvaluation  # noqa: E402
from app.evaluation.report import build_report, write_json, write_markdown  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总逐案例 TrialAgent 对照报告")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--batches", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    grouped: dict[str, list[CaseEvaluation]] = {}
    sources = sorted(args.batches.glob("batch-*/report.json"))
    if not sources:
        raise ValueError("未找到批次报告")
    seen: set[tuple[str, str]] = set()
    for source in sources:
        payload = json.loads(source.read_text(encoding="utf-8"))
        for arm, records in (payload.get("cases") or {}).items():
            for record in records:
                item = CaseEvaluation.model_validate(record)
                key = (arm, item.case_id)
                if key in seen:
                    raise ValueError(f"发现重复批次结果：{arm}/{item.case_id}")
                seen.add(key)
                grouped.setdefault(arm, []).append(item)
    report = build_report(
        dataset_version="trial-agent-locked-v1",
        dataset_sha256=dataset_sha256(args.cases),
        model_id=get_settings().qwen_model,
        prompt_version=TrialAgent.PROMPT_VERSION,
        cases_by_arm=grouped,
        metadata={"mode": "live-resumable", "batch_count": len(sources)},
    )
    write_json(report, args.output_dir / "report.json")
    write_markdown(report, args.output_dir / "report.md")
    print(f"已汇总 {len(sources)} 个批次：{args.output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
