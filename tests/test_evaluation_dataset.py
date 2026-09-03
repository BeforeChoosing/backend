import json

from app.evaluation.dataset import (
    SftRecord,
    dataset_sha256,
    split_sft_records,
    write_sft_splits,
)


def _record(case_id: str, task_id: str) -> SftRecord:
    return SftRecord(
        case_id=case_id,
        task_id=task_id,
        messages=[
            {"role": "system", "content": "只输出 JSON。"},
            {"role": "user", "content": f"任务 {task_id} 的作答"},
            {"role": "assistant", "content": '{"summary":"已完成"}'},
        ],
        metadata={"source": "human_review"},
    )


def test_split_keeps_holdout_tasks_out_of_training_and_validation(tmp_path) -> None:
    records = [_record(f"case-{index}", task_id) for index, task_id in enumerate(
        ["F-01", "F-01", "F-02", "A-01", "M-01"]
    )]

    splits = split_sft_records(records, holdout_task_ids={"M-01"}, validation_ratio=0.5)

    assert {item.task_id for item in splits["test_locked"]} == {"M-01"}
    assert not {item.task_id for item in splits["train"]} & {"M-01"}
    assert not {item.task_id for item in splits["validation"]} & {"M-01"}
    paths = write_sft_splits(splits, tmp_path)
    assert set(paths) == {"train", "validation", "test_locked", "manifest"}
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["holdout_task_ids"] == ["M-01"]
    assert dataset_sha256(paths["train"])
