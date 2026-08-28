from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from app.evaluation.metrics import summarize_cases
from app.evaluation.models import (
    ArmSummary,
    CaseEvaluation,
    EvaluationReport,
    TrialArm,
)


def build_report(
    *,
    dataset_version: str,
    dataset_sha256: str,
    model_id: str,
    prompt_version: str,
    cases_by_arm: dict[str, Iterable[CaseEvaluation]],
    metadata: dict | None = None,
) -> EvaluationReport:
    normalized_cases: dict[TrialArm, list[CaseEvaluation]] = {}
    summaries: list[ArmSummary] = []
    for arm, cases in cases_by_arm.items():
        if arm not in {"base_qwen", "prompt_hardened", "sft", "sft_validator"}:
            raise ValueError(f"unknown evaluation arm: {arm}")
        case_list = list(cases)
        normalized_cases[arm] = case_list  # type: ignore[index]
        summaries.append(ArmSummary.model_validate(summarize_cases(arm, case_list)))
    return EvaluationReport(
        dataset_version=dataset_version,
        dataset_sha256=dataset_sha256,
        model_id=model_id,
        prompt_version=prompt_version,
        arms=summaries,
        cases=normalized_cases,
        metadata=metadata or {},
    )


def write_json(report: EvaluationReport, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def render_markdown(report: EvaluationReport) -> str:
    lines = [
        "# TrialAgent 统一评测报告",
        "",
        f"- report_version: `{report.report_version}`",
        f"- dataset_version: `{report.dataset_version}`",
        f"- dataset_sha256: `{report.dataset_sha256}`",
        f"- model_id: `{report.model_id}`",
        f"- prompt_version: `{report.prompt_version}`",
        f"- generated_at: `{report.generated_at.isoformat()}`",
        "",
        "## 对照结果",
        "",
        "| 方案 | 样本数 | 结构合法率 | 分项分数 MAE | 等级准确率 | 等级 ±1 | 证据精确率 | 证据召回率 | 无效引用/样本 | 校验触发率 | 平均 API 调用 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in report.arms:
        lines.append(
            "| {arm} | {count} | {valid:.1%} | {mae} | {exact} | {within} | "
            "{precision} | {recall} | {invalid} | {trigger} | {calls} |".format(
                arm=summary.arm,
                count=summary.case_count,
                valid=summary.valid_schema_rate,
                mae=_format_number(summary.dimension_score_mae),
                exact=_format_rate(summary.level_exact_rate),
                within=_format_rate(summary.level_within_one_rate),
                precision=_format_rate(summary.evidence_precision),
                recall=_format_rate(summary.evidence_recall),
                invalid=_format_number(summary.invalid_evidence_ref_rate),
                trigger=_format_rate(summary.verifier_trigger_rate),
                calls=_format_number(summary.mean_api_calls),
            )
        )
    lines.extend(["", "## 运行说明", ""])
    if report.metadata:
        lines.append("```json")
        lines.append(json.dumps(report.metadata, ensure_ascii=False, indent=2))
        lines.append("```")
    else:
        lines.append("本报告未附加运行元数据。")
    return "\n".join(lines) + "\n"


def write_markdown(report: EvaluationReport, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report), encoding="utf-8")
    return path


def _format_rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _format_number(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"
