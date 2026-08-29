"""Combine TrialAgent, RAG, multimodal and runtime usage metrics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"报告必须是 JSON 对象：{path}")
    return value


def build_unified_report(
    *,
    trial: dict[str, Any] | None = None,
    rag: dict[str, Any] | None = None,
    multimodal: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
    input_price_per_million: float = 0.0,
    output_price_per_million: float = 0.0,
) -> dict[str, Any]:
    usage = usage or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    estimated_cost = (
        input_tokens * input_price_per_million + output_tokens * output_price_per_million
    ) / 1_000_000
    return {
        "report_version": "unified-eval-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sections": {"trial": trial, "rag": rag, "multimodal": multimodal, "usage": usage},
        "cost": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_price_per_million": input_price_per_million,
            "output_price_per_million": output_price_per_million,
            "estimated_cost": round(estimated_cost, 8),
            "note": "价格由命令行参数提供；百炼未返回 usage 时仅统计调用次数与延迟。",
        },
    }


def render_unified_markdown(report: dict[str, Any]) -> str:
    sections = report.get("sections", {})
    usage = sections.get("usage") or {}
    lines = ["# 统一评测报告", "", f"生成时间：`{report.get('generated_at', '')}`", ""]
    trial = sections.get("trial")
    if trial:
        lines += ["## TrialAgent 评价", "", "详见嵌入的 trial 报告指标。", ""]
    rag = sections.get("rag")
    if rag:
        before = rag.get("before", {})
        after = rag.get("after", {})
        lines += ["## RAG 检索", "", "| 管线 | Hit@K | MRR@K |", "|---|---:|---:|"]
        lines.append(f"| FTS5 | {before.get('hit_at_k', '—')} | {before.get('mrr_at_k', '—')} |")
        lines.append(f"| Embedding + Rerank | {after.get('hit_at_k', '—')} | {after.get('mrr_at_k', '—')} |")
        lines.append("")
    multimodal = sections.get("multimodal")
    if multimodal:
        metrics = multimodal.get("metrics", multimodal)
        lines += ["## 多模态证据定位", "", "| 指标 | 值 |", "|---|---:|"]
        for key in ("page_hit_rate", "localization_iou", "evidence_precision", "evidence_recall", "material_coverage"):
            if key in metrics:
                lines.append(f"| {key} | {metrics[key]} |")
        lines.append("")
    lines += ["## 调用成本与延迟", "", "| 指标 | 值 |", "|---|---:|"]
    for key in ("event_count", "model_call_count", "input_tokens", "output_tokens", "mean_duration_ms", "model_mean_duration_ms"):
        if key in usage:
            lines.append(f"| {key} | {usage[key]} |")
    cost = report.get("cost", {})
    lines.append(f"| estimated_cost | {cost.get('estimated_cost', 0)} |")
    lines += ["", "> 离线评测脚本不调用百炼；正式模式请求由后端审计表记录。", ""]
    return "\n".join(lines)
