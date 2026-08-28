"""Validate Sol pair reviews and emit explicit DPO pair candidates.

This command is deliberately offline.  It accepts only reviewer files written
for an existing packet, validates every evidence reference against that case,
and refuses to fabricate a rejected response or silently downgrade an invalid
pair into training data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.schemas.trial import TrialEvaluation  # noqa: E402
from app.training.cases import TrialCaseInput, load_case_inputs  # noqa: E402
from app.training.export import read_jsonl, to_dpo_record, write_jsonl  # noqa: E402
from app.training.validation import validate_evaluation  # noqa: E402
from app.tasks.catalog import get_task_definition  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验 Sol 对比结果并生成 DPO pair")
    parser.add_argument("--cases", required=True, type=Path, help="案例输入 JSONL")
    parser.add_argument("--teacher", required=True, type=Path, help="强化版评价 JSONL")
    parser.add_argument("--baseline", required=True, type=Path, help="基础版评价 JSONL")
    parser.add_argument(
        "--reviews-dir",
        required=True,
        type=Path,
        help="Sol 子任务逐案例结果目录",
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=Path("datasets/trial_agent/v1/sol_pair_reviews.local.jsonl"),
    )
    parser.add_argument(
        "--dpo-output",
        type=Path,
        default=Path("datasets/trial_agent/v1/sol_dpo_pairs.local.jsonl"),
    )
    parser.add_argument("--prompt-version", default="trial-teacher-v1")
    return parser.parse_args()


def _safe_name(case_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id)


def _by_case(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["case_id"]): row for row in read_jsonl(path)}


def _load_review(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _evaluation(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return TrialEvaluation.model_validate(value).model_dump(mode="json")
    except Exception:
        return None


def _source_evaluation(
    source: str,
    *,
    baseline: dict[str, Any],
    teacher: dict[str, Any],
    enhanced: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if source == "baseline":
        return _evaluation(baseline.get("evaluation"))
    if source == "teacher":
        return _evaluation(teacher.get("evaluation"))
    if source == "enhanced":
        return enhanced
    return None


def main() -> int:
    args = parse_args()
    try:
        cases = load_case_inputs(args.cases)
        baseline_rows = _by_case(args.baseline)
        teacher_rows = _by_case(args.teacher)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    review_rows: list[dict[str, Any]] = []
    dpo_rows: list[dict[str, Any]] = []
    counts = {"reviewed": 0, "pair_valid": 0, "exported": 0, "rejected": 0, "missing": 0}
    for index, case in enumerate(cases, 1):
        baseline = baseline_rows.get(case.case_id)
        teacher = teacher_rows.get(case.case_id)
        review = _load_review(args.reviews_dir / f"{_safe_name(case.case_id)}.json")
        if baseline is None or teacher is None or review is None:
            counts["missing"] += 1
            continue
        counts["reviewed"] += 1
        enhanced = _evaluation(review.get("enhanced_evaluation"))
        chosen_source = str(review.get("chosen_source") or "none")
        rejected_source = str(review.get("rejected_source") or "none")
        chosen = _source_evaluation(
            chosen_source,
            baseline=baseline,
            teacher=teacher,
            enhanced=enhanced,
        )
        rejected = _source_evaluation(
            rejected_source,
            baseline=baseline,
            teacher=teacher,
            enhanced=None,
        )
        valid_refs = [item["id"] for item in case.evidence_catalog if isinstance(item.get("id"), str)]
        chosen_validation = (
            validate_evaluation(get_task_definition(case.task_id), case.answer, chosen, valid_evidence_refs=valid_refs)
            if chosen is not None
            else None
        )
        rejected_validation = (
            validate_evaluation(get_task_definition(case.task_id), case.answer, rejected, valid_evidence_refs=valid_refs)
            if rejected is not None
            else None
        )
        pair_valid = bool(review.get("pair_valid"))
        if (
            not pair_valid
            or chosen is None
            or rejected is None
            or chosen_source == rejected_source
            or chosen_validation is None
            or chosen_validation.status == "rejected"
            or rejected_validation is None
            or chosen == rejected
        ):
            counts["rejected"] += 1
            pair_valid = False
        else:
            counts["pair_valid"] += 1
        review_row = {
            "case_id": case.case_id,
            "task_id": case.task_id,
            "pair_valid": pair_valid,
            "chosen_source": chosen_source,
            "rejected_source": rejected_source,
            "enhanced_evaluation": enhanced,
            "rationale": str(review.get("rationale") or ""),
            "issues": review.get("issues") if isinstance(review.get("issues"), list) else [],
            "chosen_validation": chosen_validation.as_dict() if chosen_validation else None,
            "rejected_validation": rejected_validation.as_dict() if rejected_validation else None,
            "review_model": "gpt-5.6-sol",
            "reasoning_effort": "high",
        }
        review_rows.append(review_row)
        if pair_valid and chosen is not None and rejected is not None:
            pair_row = {
                **case.as_record(),
                "chosen_evaluation": chosen,
                "rejected_evaluation": rejected,
                "metadata": {
                    **dict(case.metadata),
                    "source": "sol_pair_review",
                    "human_review_required": True,
                    "sol_review_model": "gpt-5.6-sol",
                    "sol_review_reasoning": "high",
                    "sol_rationale": str(review.get("rationale") or ""),
                    "chosen_source": chosen_source,
                    "rejected_source": rejected_source,
                },
            }
            dpo_record = to_dpo_record(pair_row, index=index, prompt_version=args.prompt_version)
            if dpo_record is not None:
                dpo_rows.append(dpo_record)
                counts["exported"] += 1
    write_jsonl(args.review_output, review_rows)
    write_jsonl(args.dpo_output, dpo_rows)
    print(json.dumps({**counts, "review_output": str(args.review_output), "dpo_output": str(args.dpo_output)}, ensure_ascii=False))
    return 0 if counts["missing"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
