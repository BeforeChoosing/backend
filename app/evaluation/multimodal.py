from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


def _validate_bbox(values: list[int]) -> list[int]:
    if len(values) != 4 or any(value < 0 or value > 1000 for value in values):
        raise ValueError("bbox 必须是 0–1000 的四个坐标")
    x0, y0, x1, y1 = values
    if x1 <= x0 or y1 <= y0:
        raise ValueError("bbox 必须满足右下角大于左上角")
    return values


class MultimodalGoldEvidence(BaseModel):
    """Human-reviewed region used as the localization reference."""

    evidence_id: str = Field(min_length=1, max_length=160)
    material_id: str = Field(min_length=1, max_length=120)
    page: int = Field(ge=1)
    bbox: list[int] = Field(min_length=4, max_length=4)
    quote: str = Field(min_length=1, max_length=800)
    required: bool = True

    @model_validator(mode="after")
    def validate_region(self) -> "MultimodalGoldEvidence":
        _validate_bbox(self.bbox)
        return self


class MultimodalEvaluationCase(BaseModel):
    """A locked material bundle and its manually reviewed evidence regions."""

    case_id: str = Field(min_length=1, max_length=120)
    materials: list[str] = Field(min_length=1, max_length=40)
    gold: list[MultimodalGoldEvidence] = Field(default_factory=list, max_length=160)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_materials(self) -> "MultimodalEvaluationCase":
        material_ids = set(self.materials)
        if len(material_ids) != len(self.materials):
            raise ValueError("materials 不能包含重复 material_id")
        unknown = {item.material_id for item in self.gold} - material_ids
        if unknown:
            raise ValueError(f"金标准包含未登记的材料：{sorted(unknown)}")
        evidence_ids = [item.evidence_id for item in self.gold]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("gold evidence_id 不能重复")
        return self


class MultimodalPredictionItem(BaseModel):
    """One model prediction exported with the material it came from."""

    material_id: str = Field(min_length=1, max_length=120)
    page: int = Field(ge=1)
    bbox: list[int] = Field(min_length=4, max_length=4)
    quote: str = Field(default="", max_length=800)
    confidence: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_region(self) -> "MultimodalPredictionItem":
        _validate_bbox(self.bbox)
        return self


class MultimodalCaseEvaluation(BaseModel):
    case_id: str
    schema_valid: bool = True
    gold_evidence_count: int = Field(default=0, ge=0)
    predicted_evidence_count: int = Field(default=0, ge=0)
    matched_evidence_count: int = Field(default=0, ge=0)
    page_hit_rate: float = Field(default=0.0, ge=0, le=1)
    localization_iou: float | None = Field(default=None, ge=0, le=1)
    evidence_precision: float = Field(default=0.0, ge=0, le=1)
    evidence_recall: float = Field(default=0.0, ge=0, le=1)
    evidence_f1: float = Field(default=0.0, ge=0, le=1)
    material_coverage: float = Field(default=0.0, ge=0, le=1)
    page_coverage: float = Field(default=0.0, ge=0, le=1)
    invalid_prediction_count: int = Field(default=0, ge=0)
    api_calls: int = Field(default=0, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)


class MultimodalSummary(BaseModel):
    case_count: int = Field(ge=0)
    page_hit_rate: float = Field(ge=0, le=1)
    localization_iou: float | None = Field(default=None, ge=0, le=1)
    evidence_precision: float = Field(ge=0, le=1)
    evidence_recall: float = Field(ge=0, le=1)
    evidence_f1: float = Field(ge=0, le=1)
    material_coverage: float = Field(ge=0, le=1)
    page_coverage: float = Field(ge=0, le=1)
    invalid_prediction_rate: float = Field(ge=0)
    mean_api_calls: float = Field(ge=0)
    mean_latency_ms: float | None = Field(default=None, ge=0)


class MultimodalEvaluationReport(BaseModel):
    report_version: str = "multimodal-eval-v1"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dataset_version: str
    dataset_sha256: str
    model_id: str
    summary: MultimodalSummary
    cases: list[MultimodalCaseEvaluation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _bbox_iou(left: list[int], right: list[int]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _normalize_quote(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _quote_similarity(left: str, right: str) -> float:
    normalized_left = _normalize_quote(left)
    normalized_right = _normalize_quote(right)
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left in normalized_right or normalized_right in normalized_left:
        return 1.0
    return SequenceMatcher(None, normalized_left, normalized_right).ratio()


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_multimodal_case(
    case: MultimodalEvaluationCase,
    predictions: Iterable[MultimodalPredictionItem | dict[str, Any]],
    *,
    iou_threshold: float = 0.5,
    quote_threshold: float = 0.5,
    api_calls: int = 0,
    latency_ms: float | None = None,
) -> MultimodalCaseEvaluation:
    """Match predictions one-to-one and calculate localization/coverage metrics."""

    if not 0 <= iou_threshold <= 1 or not 0 <= quote_threshold <= 1:
        raise ValueError("匹配阈值必须位于 [0, 1] 区间")
    parsed_predictions: list[MultimodalPredictionItem] = []
    invalid_count = 0
    for raw in predictions:
        try:
            parsed_predictions.append(
                raw if isinstance(raw, MultimodalPredictionItem)
                else MultimodalPredictionItem.model_validate(raw)
            )
        except (TypeError, ValueError):
            invalid_count += 1

    required_gold = [item for item in case.gold if item.required]
    unmatched_gold = set(range(len(required_gold)))
    matches: list[tuple[int, float]] = []
    for prediction in parsed_predictions:
        candidates: list[tuple[float, int]] = []
        for index in unmatched_gold:
            gold = required_gold[index]
            if gold.material_id != prediction.material_id or gold.page != prediction.page:
                continue
            iou = _bbox_iou(gold.bbox, prediction.bbox)
            if iou < iou_threshold or _quote_similarity(gold.quote, prediction.quote) < quote_threshold:
                continue
            candidates.append((iou, index))
        if not candidates:
            continue
        iou, index = max(candidates)
        unmatched_gold.remove(index)
        matches.append((index, iou))

    gold_count = len(required_gold)
    predicted_count = len(parsed_predictions)
    matched_count = len(matches)
    precision = matched_count / predicted_count if predicted_count else 0.0
    recall = matched_count / gold_count if gold_count else 1.0
    gold_materials = {item.material_id for item in required_gold}
    matched_materials = {required_gold[index].material_id for index, _ in matches}
    gold_pages = {(item.material_id, item.page) for item in required_gold}
    matched_pages = {(required_gold[index].material_id, required_gold[index].page) for index, _ in matches}
    return MultimodalCaseEvaluation(
        case_id=case.case_id,
        schema_valid=invalid_count == 0,
        gold_evidence_count=gold_count,
        predicted_evidence_count=predicted_count,
        matched_evidence_count=matched_count,
        page_hit_rate=recall,
        localization_iou=(sum(iou for _, iou in matches) / matched_count if matches else None),
        evidence_precision=precision,
        evidence_recall=recall,
        evidence_f1=_f1(precision, recall),
        material_coverage=(len(matched_materials) / len(gold_materials) if gold_materials else 1.0),
        page_coverage=(len(matched_pages) / len(gold_pages) if gold_pages else 1.0),
        invalid_prediction_count=invalid_count,
        api_calls=api_calls,
        latency_ms=latency_ms,
    )


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def build_multimodal_report(
    *,
    dataset_version: str,
    dataset_sha256: str,
    model_id: str,
    cases: Iterable[MultimodalCaseEvaluation],
    metadata: dict[str, Any] | None = None,
) -> MultimodalEvaluationReport:
    case_list = list(cases)
    valid = [case for case in case_list if case.schema_valid]
    ious = [case.localization_iou for case in valid if case.localization_iou is not None]
    summary = MultimodalSummary(
        case_count=len(case_list),
        page_hit_rate=_mean(case.page_hit_rate for case in valid),
        localization_iou=(_mean(ious) if ious else None),
        evidence_precision=_mean(case.evidence_precision for case in valid),
        evidence_recall=_mean(case.evidence_recall for case in valid),
        evidence_f1=_mean(case.evidence_f1 for case in valid),
        material_coverage=_mean(case.material_coverage for case in valid),
        page_coverage=_mean(case.page_coverage for case in valid),
        invalid_prediction_rate=(
            sum(case.invalid_prediction_count for case in case_list) / len(case_list)
            if case_list else 0.0
        ),
        mean_api_calls=_mean(float(case.api_calls) for case in case_list),
        mean_latency_ms=(
            _mean(case.latency_ms for case in case_list if case.latency_ms is not None)
            if any(case.latency_ms is not None for case in case_list)
            else None
        ),
    )
    return MultimodalEvaluationReport(
        dataset_version=dataset_version,
        dataset_sha256=dataset_sha256,
        model_id=model_id,
        summary=summary,
        cases=case_list,
        metadata=metadata or {},
    )


def render_multimodal_markdown(report: MultimodalEvaluationReport) -> str:
    summary = report.summary
    lines = [
        "# 多模态证据定位评测报告",
        "",
        f"- report_version: `{report.report_version}`",
        f"- dataset_version: `{report.dataset_version}`",
        f"- dataset_sha256: `{report.dataset_sha256}`",
        f"- model_id: `{report.model_id}`",
        f"- generated_at: `{report.generated_at.isoformat()}`",
        "",
        "| 样本数 | 页码命中率 | 定位 IoU | 证据精确率 | 证据召回率 | 证据 F1 | 材料覆盖率 | 页面覆盖率 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {summary.case_count} | {summary.page_hit_rate:.1%} | "
            f"{_format_rate(summary.localization_iou)} | {summary.evidence_precision:.1%} | "
            f"{summary.evidence_recall:.1%} | {summary.evidence_f1:.1%} | "
            f"{summary.material_coverage:.1%} | {summary.page_coverage:.1%} |"
        ),
        "",
        "## 运行元数据",
        "",
        "```json",
        json.dumps(report.metadata, ensure_ascii=False, indent=2),
        "```",
    ]
    return "\n".join(lines) + "\n"


def _format_rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def write_multimodal_json(report: MultimodalEvaluationReport, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def write_multimodal_markdown(report: MultimodalEvaluationReport, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_multimodal_markdown(report), encoding="utf-8")
    return path
