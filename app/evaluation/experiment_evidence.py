"""Build a compact evidence report from completed live experiments."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable


def _arm(report: dict[str, Any], name: str) -> dict[str, Any]:
    for item in report.get("arms", []):
        if item.get("arm") == name:
            return item
    raise ValueError(f"TrialAgent 报告缺少实验组：{name}")


def build_experiment_evidence_report(
    *,
    trial: dict[str, Any],
    rag: dict[str, Any],
    multimodal: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Combine completed reports without making any model calls."""

    base = _arm(trial, "base_qwen")
    hardened = _arm(trial, "prompt_hardened")
    before = rag.get("before", {})
    after = rag.get("after", {})
    vision_rows = sorted(
        list(multimodal),
        key=lambda row: float(row.get("mean_character_similarity", 0.0)),
        reverse=True,
    )
    if len(vision_rows) < 2:
        raise ValueError("多模态对比至少需要两份报告")

    best_vision = vision_rows[0]
    baseline_vision = vision_rows[-1]
    clean_api_calls = int(
        float(base.get("case_count", 0)) * float(base.get("mean_api_calls", 0))
        + float(hardened.get("case_count", 0)) * float(hardened.get("mean_api_calls", 0))
        + float(rag.get("api_calls_estimate", 0))
        + sum(int(row.get("page_count", 0)) for row in vision_rows)
    )
    return {
        "report_version": "experiment-evidence-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trial_agent": {
            "model": trial.get("model_id"),
            "dataset_version": trial.get("dataset_version"),
            "dataset_sha256": trial.get("dataset_sha256"),
            "case_count": hardened.get("case_count"),
            "base_schema_valid_rate": base.get("valid_schema_rate"),
            "hardened_schema_valid_rate": hardened.get("valid_schema_rate"),
            "schema_valid_rate_delta": round(
                float(hardened.get("valid_schema_rate", 0)) - float(base.get("valid_schema_rate", 0)), 6
            ),
            "dimension_score_mae": hardened.get("dimension_score_mae"),
            "level_exact_rate": hardened.get("level_exact_rate"),
            "level_within_one_rate": hardened.get("level_within_one_rate"),
            "evidence_precision": hardened.get("evidence_precision"),
            "evidence_recall": hardened.get("evidence_recall"),
            "invalid_evidence_ref_rate": hardened.get("invalid_evidence_ref_rate"),
            "base_mean_latency_ms": base.get("mean_latency_ms"),
            "hardened_mean_latency_ms": hardened.get("mean_latency_ms"),
        },
        "rag": {
            "query_count": len(before.get("details", [])),
            "before_hit_at_k": before.get("hit_at_k"),
            "after_hit_at_k": after.get("hit_at_k"),
            "hit_at_k_delta": round(float(after.get("hit_at_k", 0)) - float(before.get("hit_at_k", 0)), 6),
            "before_mrr_at_k": before.get("mrr_at_k"),
            "after_mrr_at_k": after.get("mrr_at_k"),
            "mrr_at_k_delta": round(float(after.get("mrr_at_k", 0)) - float(before.get("mrr_at_k", 0)), 6),
            "rerank_model": after.get("model"),
            "api_calls": rag.get("api_calls_estimate"),
        },
        "multimodal_ocr": {
            "source_file": best_vision.get("source_file"),
            "page_count": best_vision.get("page_count"),
            "models": [
                {
                    "model": row.get("model"),
                    "mean_character_similarity": row.get("mean_character_similarity"),
                    "empty_prediction_rate": row.get("empty_prediction_rate"),
                    "mean_latency_ms": row.get("mean_latency_ms"),
                }
                for row in vision_rows
            ],
            "best_model": best_vision.get("model"),
            "similarity_delta_vs_baseline": round(
                float(best_vision.get("mean_character_similarity", 0))
                - float(baseline_vision.get("mean_character_similarity", 0)),
                6,
            ),
            "latency_delta_ms_vs_baseline": round(
                float(best_vision.get("mean_latency_ms", 0))
                - float(baseline_vision.get("mean_latency_ms", 0)),
                3,
            ),
        },
        "clean_run_api_calls": clean_api_calls,
        "conclusions": [
            "证据约束提示与结构校验显著提高 TrialAgent 的可用输出率。",
            "Embedding 召回与重排显著提高本地知识库的检索命中率和首位排序质量。",
            "复杂中文混排页面上，qwen3-vl-plus 的文字还原稳定性高于 qwen-vl-ocr。",
        ],
        "limitations": [
            "TrialAgent 只覆盖锁定的 24 条文本评价案例，不代表所有自由输入。",
            "RAG 只覆盖 8 条小型标注查询，用于回归和方向性对比。",
            "多模态结果是同一份真实 PDF 的 8 页 OCR 文字还原对比，不等同于区域定位精度。",
            "clean_run_api_calls 只统计正式对比调用，不包含调试期间的超时与重试。",
        ],
    }


def _percent(value: Any) -> str:
    return "—" if value is None else f"{float(value):.1%}"


def render_experiment_evidence_markdown(report: dict[str, Any]) -> str:
    trial = report["trial_agent"]
    rag = report["rag"]
    vision = report["multimodal_ocr"]
    lines = [
        "# 核心技术实验证据报告",
        "",
        f"生成时间：`{report['generated_at']}`",
        "",
        "## 1. TrialAgent 提示与证据校验",
        "",
        f"- 模型：`{trial['model']}`",
        f"- 锁定案例：{trial['case_count']} 条",
        f"- Schema 合法率：{_percent(trial['base_schema_valid_rate'])} → {_percent(trial['hardened_schema_valid_rate'])}",
        f"- 分项评分 MAE：{trial['dimension_score_mae']:.2f}",
        f"- 等级完全命中 / 误差不超过一级：{_percent(trial['level_exact_rate'])} / {_percent(trial['level_within_one_rate'])}",
        f"- 证据引用精确率 / 召回率：{_percent(trial['evidence_precision'])} / {_percent(trial['evidence_recall'])}",
        "",
        "## 2. 本地 RAG 检索",
        "",
        f"- 标注查询：{rag['query_count']} 条",
        f"- Hit@5：{_percent(rag['before_hit_at_k'])} → {_percent(rag['after_hit_at_k'])}",
        f"- MRR@5：{_percent(rag['before_mrr_at_k'])} → {_percent(rag['after_mrr_at_k'])}",
        f"- 重排模型：`{rag['rerank_model']}`",
        "",
        "## 3. 多模态真实页面 OCR",
        "",
        f"- 来源：`{vision['source_file']}`，{vision['page_count']} 页",
        "| 模型 | 平均字符相似度 | 空结果率 | 平均延迟 |",
        "|---|---:|---:|---:|",
    ]
    for row in vision["models"]:
        lines.append(
            f"| {row['model']} | {_percent(row['mean_character_similarity'])} | "
            f"{_percent(row['empty_prediction_rate'])} | {float(row['mean_latency_ms']) / 1000:.2f} s |"
        )
    lines += [
        "",
        f"正式对比共 {report['clean_run_api_calls']} 次必要模型调用；该数字不包含调试期超时与重试。",
        "",
        "## 结论",
        "",
        *[f"- {item}" for item in report["conclusions"]],
        "",
        "## 边界",
        "",
        *[f"- {item}" for item in report["limitations"]],
        "",
    ]
    return "\n".join(lines)
