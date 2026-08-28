"""Generate cached teacher labels for text-only TrialAgent cases.

The command never fabricates a label when the model is unavailable. Use
``--dry-run`` to inspect the planned calls without contacting Bailian.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.config import get_settings  # noqa: E402
from app.services.llm_gateway import LLMGatewayError  # noqa: E402
from app.tasks.catalog import get_task_definition  # noqa: E402
from app.training.cases import TrialCaseInput, load_case_inputs  # noqa: E402
from app.training.cases import case_fingerprint  # noqa: E402
from app.training.teacher import (  # noqa: E402
    TeacherCache,
    TeacherLabeler,
    TeacherResponse,
    response_fingerprint,
)
from app.training.validation import ValidationResult, validate_evaluation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用百炼教师模型生成并校验 TrialAgent 银标")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="原始案例 JSONL；每行包含 case_id、task_id 和 answer",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/trial_agent/v1/teacher_labels.local.jsonl"),
        help="教师评价输出 JSONL，默认写入本地忽略路径",
    )
    parser.add_argument("--model", help="教师模型；默认读取 TRIAL_TEACHER_MODEL")
    parser.add_argument("--review-model", help="低置信度复核模型；默认读取 TRIAL_REVIEW_MODEL")
    parser.add_argument("--no-review", action="store_true", help="不升级到复核模型")
    parser.add_argument("--cache", type=Path, help="SQLite 缓存路径；默认读取 TRIAL_TEACHER_CACHE_PATH")
    parser.add_argument("--prompt-version", help="教师 Prompt 版本；默认读取 TRIAL_TEACHER_PROMPT_VERSION")
    parser.add_argument("--limit", type=int, help="最多处理多少条案例")
    parser.add_argument("--resume", action="store_true", help="跳过输出文件中已有的 case_id")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重新调用教师模型")
    parser.add_argument("--dry-run", action="store_true", help="只检查输入并显示计划，不调用百炼")
    return parser.parse_args()


def _read_existing_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"输出文件 {path} 第 {line_number} 行不是有效 JSON：{exc}") from exc
        if isinstance(row, dict) and isinstance(row.get("case_id"), str):
            ids.add(row["case_id"])
    return ids


def _write_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _validation_record(
    case: TrialCaseInput,
    response: Any,
    validation: ValidationResult | None,
    *,
    review_response: Any | None = None,
) -> dict[str, Any]:
    task = get_task_definition(case.task_id)
    evidence_refs = [item["id"] for item in case.evidence_catalog if isinstance(item.get("id"), str)]
    teacher_info = {
        "model": response.model,
        "prompt_version": response.prompt_version,
        "request_fingerprint": response.fingerprint,
        "cache_hit": response.cache_hit,
        "api_calls": response.api_calls,
        "status": response.status,
    }
    if response.error:
        teacher_info["error"] = response.error
    if review_response is not None:
        teacher_info["review_model"] = review_response.model
        teacher_info["review_prompt_version"] = review_response.prompt_version
        teacher_info["review_request_fingerprint"] = review_response.fingerprint
        teacher_info["review_cache_hit"] = review_response.cache_hit
        teacher_info["review_api_calls"] = review_response.api_calls
    if validation is None:
        validation_payload: dict[str, Any] = {
            "status": "planned",
            "confidence": None,
            "schema_valid": None,
            "evidence_coverage": None,
            "reason_codes": ["dry_run"],
        }
        status = "planned"
        evaluation = None
    else:
        validation_payload = validation.as_dict()
        status = validation.status
        evaluation = validation.evaluation
    return {
        "case_id": case.case_id,
        "task_id": task.id,
        "answer": case.answer.model_dump(mode="json"),
        "confirmed_card_ids": list(case.confirmed_card_ids),
        "evidence_catalog": list(case.evidence_catalog),
        "evaluation": evaluation,
        "validation": validation_payload,
        "teacher": teacher_info,
        "metadata": {
            **dict(case.metadata),
            "source": "teacher_label_pipeline",
            "label_status": status,
            "multimodal_training": False,
        },
        "valid_evidence_refs": evidence_refs,
    }


def _duplicate_input_record(
    case: TrialCaseInput,
    *,
    model: str,
    prompt_version: str,
    duplicate_of: str,
) -> dict[str, Any]:
    """Record a duplicate without spending a teacher-model request."""

    fingerprint = case_fingerprint(case, model=model, prompt_version=prompt_version)
    response = TeacherResponse(
        raw=None,
        model=model,
        prompt_version=prompt_version,
        fingerprint=fingerprint,
        cache_hit=False,
        api_calls=0,
        status="skipped_duplicate_input",
        error=f"与案例 {duplicate_of} 的输入完全相同",
    )
    record = _validation_record(case, response, None)
    record["validation"] = {
        "status": "skipped_duplicate_input",
        "confidence": None,
        "schema_valid": None,
        "evidence_coverage": None,
        "reason_codes": ["duplicate_case_input"],
        "details": {"duplicate_of": duplicate_of},
    }
    record["metadata"]["label_status"] = "skipped_duplicate_input"
    record["metadata"]["duplicate_of"] = duplicate_of
    return record


def _mark_duplicate_label(
    record: dict[str, Any],
    *,
    seen: dict[str, str],
) -> dict[str, Any]:
    """Route byte-identical evaluations to review instead of training on them."""

    evaluation = record.get("evaluation")
    if not isinstance(evaluation, dict):
        return record
    fingerprint = response_fingerprint(evaluation)
    case_id = str(record.get("case_id", ""))
    previous = seen.get(fingerprint)
    if previous is None:
        seen[fingerprint] = case_id
        return record
    validation = record.get("validation")
    if isinstance(validation, dict):
        reason_codes = list(validation.get("reason_codes") or [])
        if "duplicate_teacher_response" not in reason_codes:
            reason_codes.append("duplicate_teacher_response")
        validation["reason_codes"] = reason_codes
        validation["status"] = "needs_review"
        current_confidence = validation.get("confidence")
        if isinstance(current_confidence, (int, float)):
            validation["confidence"] = min(float(current_confidence), 0.5)
        details = validation.setdefault("details", {})
        if isinstance(details, dict):
            details["duplicate_of"] = previous
    metadata = record.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["label_status"] = "needs_review"
        metadata["duplicate_of"] = previous
    return record


def _pick_validation(
    first: ValidationResult,
    second: ValidationResult,
) -> ValidationResult:
    """Prefer a schema-valid result with more evidence and fewer violations."""

    if first.status == "rejected" and second.status != "rejected":
        return second
    if second.status == "rejected" and first.status != "rejected":
        return first
    if second.confidence > first.confidence:
        return second
    return first


def _label_case(
    labeler: TeacherLabeler,
    case: TrialCaseInput,
    *,
    model: str,
    prompt_version: str,
    review_model: str | None,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    response = labeler.label(
        case,
        model=model,
        prompt_version=prompt_version,
        force=force,
        dry_run=dry_run,
    )
    if response.raw is None:
        return _validation_record(case, response, None)
    task = get_task_definition(case.task_id)
    valid_refs = [item["id"] for item in case.evidence_catalog if isinstance(item.get("id"), str)]
    validation = validate_evaluation(task, case.answer, response.raw, valid_evidence_refs=valid_refs)
    review_response = None
    if review_model and validation.status == "needs_review":
        review_response = labeler.label(
            case,
            model=review_model,
            prompt_version=f"{prompt_version}-review",
            force=force,
            dry_run=dry_run,
        )
        if review_response.raw is not None:
            review_validation = validate_evaluation(
                task,
                case.answer,
                review_response.raw,
                valid_evidence_refs=valid_refs,
            )
            chosen = _pick_validation(validation, review_validation)
            if chosen is review_validation:
                validation = review_validation
                response = review_response
    record = _validation_record(case, response, validation, review_response=review_response)
    if review_response is not None:
        record["metadata"]["review_model_used"] = True
    return record


def main() -> int:
    args = parse_args()
    settings = get_settings()
    try:
        cases = load_case_inputs(args.input)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.limit is not None:
        if args.limit < 1:
            print("--limit 必须大于 0", file=sys.stderr)
            return 2
        cases = cases[: args.limit]
    existing_ids = _read_existing_case_ids(args.output) if args.resume else set()
    if not args.resume and args.output.exists():
        args.output.write_text("", encoding="utf-8")
    cases = [case for case in cases if case.case_id not in existing_ids]
    if not cases:
        print("没有需要处理的案例。")
        return 0
    model = (args.model or settings.trial_teacher_model).strip()
    prompt_version = (args.prompt_version or settings.trial_teacher_prompt_version).strip()
    review_model = None if args.no_review else (args.review_model or settings.trial_review_model).strip()
    cache_path = args.cache or Path(settings.trial_teacher_cache_path)
    labeler = TeacherLabeler(settings=settings, cache=TeacherCache(cache_path))

    planned_calls = len(cases) * (1 + (1 if review_model else 0))
    print(f"案例数：{len(cases)}")
    print(f"教师模型：{model}")
    print(f"复核模型：{review_model or '关闭'}")
    print(f"缓存：{cache_path}")
    print(f"最多计划调用：{planned_calls}（实际会因缓存和低置信度路由减少）")
    if args.dry_run:
        for case in cases:
            record = _label_case(
                labeler,
                case,
                model=model,
                prompt_version=prompt_version,
                review_model=None,
                force=args.force,
                dry_run=True,
            )
            _write_record(args.output, record)
        print(f"dry-run 计划已写入：{args.output}")
        return 0

    failed = 0
    seen_case_inputs: dict[str, str] = {}
    seen_labels: dict[str, str] = {}
    for index, case in enumerate(cases, 1):
        input_fingerprint = case_fingerprint(
            case,
            model="__case_input__",
            prompt_version="__case_input_v1__",
        )
        duplicate_of = seen_case_inputs.get(input_fingerprint)
        if duplicate_of is not None:
            record = _duplicate_input_record(
                case,
                model=model,
                prompt_version=prompt_version,
                duplicate_of=duplicate_of,
            )
            _write_record(args.output, record)
            print(f"[{index}/{len(cases)}] {case.case_id} -> skipped_duplicate_input（重复 {duplicate_of}）")
            continue
        seen_case_inputs[input_fingerprint] = case.case_id
        try:
            record = _label_case(
                labeler,
                case,
                model=model,
                prompt_version=prompt_version,
                review_model=review_model,
                force=args.force,
                dry_run=False,
            )
        except (LLMGatewayError, ValueError, OSError) as exc:
            failed += 1
            print(f"[{index}/{len(cases)}] {case.case_id} 失败：{exc}", file=sys.stderr)
            continue
        record = _mark_duplicate_label(record, seen=seen_labels)
        _write_record(args.output, record)
        validation = record["validation"]
        print(
            f"[{index}/{len(cases)}] {case.case_id} -> {validation.get('status')} "
            f"confidence={validation.get('confidence')} "
            f"api_calls={record['teacher']['api_calls']}"
        )
    print(f"输出：{args.output}")
    print(f"缓存条目：{labeler.cache.count()}")
    if failed:
        print(f"失败案例：{failed}，未写入伪造评价。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
