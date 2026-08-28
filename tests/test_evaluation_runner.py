import json

from app.evaluation.runner import load_prediction_records, run_offline_evaluation


def test_offline_runner_aggregates_prediction_jsonl(tmp_path) -> None:
    case_path = tmp_path / "cases.jsonl"
    case_path.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "task_id": "F-01",
                "answer": {"step_answers": {}},
                "gold": {"dimensions": {}, "observed_level": "证据不足"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    prediction_path = tmp_path / "predictions.jsonl"
    prediction_path.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "arm": "base_qwen",
                "evaluation": {
                    "summary": "证据不足。",
                    "dimensions": [],
                    "observed_level": "证据不足",
                    "confidence": "低",
                    "strengths": [],
                    "gaps": [],
                    "next_step": "补充作答。",
                },
                "valid_evidence_refs": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_prediction_records(prediction_path)
    report = run_offline_evaluation(case_path, records)

    assert report.arms[0].arm == "base_qwen"
    assert report.arms[0].case_count == 1
