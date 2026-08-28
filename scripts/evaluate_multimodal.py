"""Generate an offline Qwen-VL evidence localization report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.evaluation.dataset import dataset_sha256
from app.evaluation.multimodal import (
    MultimodalCaseEvaluation,
    MultimodalEvaluationCase,
    build_multimodal_report,
    evaluate_multimodal_case,
    write_multimodal_json,
    write_multimodal_markdown,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"无法读取 {path} 第 {line_number} 行：{exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path} 第 {line_number} 行必须是 JSON 对象")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} 不包含可用记录")
    return rows


def _load_cases(path: Path) -> list[MultimodalEvaluationCase]:
    cases: list[MultimodalEvaluationCase] = []
    for row in _read_jsonl(path):
        try:
            cases.append(MultimodalEvaluationCase.model_validate(row))
        except ValueError as exc:
            raise ValueError(f"评测集存在无效记录：{exc}") from exc
    return cases


def _load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("预测记录缺少 case_id")
        if case_id in predictions:
            raise ValueError(f"预测记录重复：{case_id}")
        predictions[case_id] = row
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="生成离线多模态证据定位评测报告")
    parser.add_argument("--cases", required=True, type=Path, help="人工标注 JSONL")
    parser.add_argument("--predictions", required=True, type=Path, help="Qwen-VL 预测 JSONL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation-results/multimodal-v1"),
        help="报告输出目录（已在 .gitignore 中忽略）",
    )
    parser.add_argument("--model-id", default="qwen-vl-ocr", help="预测使用的视觉模型 ID")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--quote-threshold", type=float, default=0.5)
    args = parser.parse_args()

    cases = _load_cases(args.cases)
    predictions = _load_predictions(args.predictions)
    case_ids = {case.case_id for case in cases}
    unknown = set(predictions) - case_ids
    missing = case_ids - set(predictions)
    if unknown:
        raise ValueError(f"预测记录引用了评测集之外的 case_id：{sorted(unknown)}")
    if missing:
        raise ValueError(f"预测记录缺少 case_id：{sorted(missing)}")

    evaluations: list[MultimodalCaseEvaluation] = []
    for case in cases:
        prediction = predictions[case.case_id]
        items = prediction.get("items", [])
        if not isinstance(items, list):
            raise ValueError(f"{case.case_id} 的 items 必须是数组")
        evaluations.append(
            evaluate_multimodal_case(
                case,
                items,
                iou_threshold=args.iou_threshold,
                quote_threshold=args.quote_threshold,
                api_calls=int(prediction.get("api_calls", 0)),
                latency_ms=prediction.get("latency_ms"),
            )
        )

    report = build_multimodal_report(
        dataset_version="multimodal-eval-v1",
        dataset_sha256=dataset_sha256(args.cases),
        model_id=args.model_id,
        cases=evaluations,
        metadata={
            "mode": "offline",
            "iou_threshold": args.iou_threshold,
            "quote_threshold": args.quote_threshold,
        },
    )
    json_path = write_multimodal_json(report, args.output_dir / "report.json")
    markdown_path = write_multimodal_markdown(report, args.output_dir / "report.md")
    print(f"已写入：{json_path}")
    print(f"已写入：{markdown_path}")


if __name__ == "__main__":
    main()
