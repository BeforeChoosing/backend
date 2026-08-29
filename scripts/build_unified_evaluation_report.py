"""Build one report from existing offline evaluation artifacts and audit logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.evaluation.unified import build_unified_report, load_json, render_unified_markdown  # noqa: E402
from app.services.audit_log import AuditLogStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="合并统一评测报告（离线，不调用模型）")
    parser.add_argument("--trial-report", type=Path)
    parser.add_argument("--rag-report", type=Path)
    parser.add_argument("--multimodal-report", type=Path)
    parser.add_argument("--audit-db", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation-results" / "unified-v1")
    parser.add_argument("--input-price-per-million", type=float, default=0.0)
    parser.add_argument("--output-price-per-million", type=float, default=0.0)
    args = parser.parse_args()
    usage = AuditLogStore(args.audit_db or get_settings().profile_db_path).usage_summary(app_mode="use")
    report = build_unified_report(
        trial=load_json(args.trial_report), rag=load_json(args.rag_report),
        multimodal=load_json(args.multimodal_report), usage=usage,
        input_price_per_million=args.input_price_per_million,
        output_price_per_million=args.output_price_per_million,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "report.md").write_text(render_unified_markdown(report), encoding="utf-8")
    print(f"已写入：{args.output_dir / 'report.json'}")
    print(f"已写入：{args.output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
