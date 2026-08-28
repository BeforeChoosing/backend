import json
from pathlib import Path

from app.config import Settings
from app.schemas.task_catalog import DynamicTrialAnswer
from app.tasks.catalog import get_task_definition
from app.training.cases import TrialCaseInput, load_case_inputs
from app.training.export import export_dpo_records, export_sft_records
from app.training.generation import CaseGenerator
from app.training.teacher import TeacherCache, TeacherLabeler
from app.training.validation import validate_evaluation


def _case(task_id: str = "M-02", case_id: str = "case-001") -> TrialCaseInput:
    task = get_task_definition(task_id)
    answer = DynamicTrialAnswer(
        step_answers={step.id: f"针对{step.title}给出可执行方案。" for step in task.steps},
        viewed_material_ids=[material.id for material in task.materials[:2]],
        evidence_refs=[material.id for material in task.materials[:2]],
        event_decision="调整",
        event_response="提高高风险样本权重并重跑分层评测。",
    )
    catalog = [
        {"id": f"answer:{step.id}", "content": "作答证据"}
        for step in task.steps
    ]
    catalog.extend(
        {"id": f"material:{material.id}", "content": "材料证据"}
        for material in task.materials[:2]
    )
    catalog.extend(
        [
            {"id": "event:decision", "content": "事件处理决定"},
            {"id": "event:response", "content": "事件处理依据"},
        ]
    )
    return TrialCaseInput(
        case_id=case_id,
        task_id=task.id,
        answer=answer,
        evidence_catalog=tuple(catalog),
    )


def _raw_evaluation(case: TrialCaseInput, *, confidence: str = "高") -> dict:
    task = get_task_definition(case.task_id)
    refs = [item["id"] for item in case.evidence_catalog]
    dimensions = [
        {
            "dimension": criterion.dimension,
            "weight": criterion.weight,
            "score": 80,
            "evidence": f"引用 {criterion.dimension} 的作答行为。",
            "evidence_refs": [refs[index % len(refs)]],
        }
        for index, criterion in enumerate(task.rubric)
    ]
    return {
        "summary": "能按证据说明判断，并在事件后调整方案。",
        "dimensions": dimensions,
        "primary_ability": "错误能力名会由服务端覆盖",
        "observed_level": "L3",
        "level_reason": "五步作答和事件响应均有可核对证据。",
        "supporting_evidence": [],
        "process_evidence": ["完成五步作答"],
        "coach_dependency": "独立完成",
        "strengths": ["证据引用清楚"],
        "gaps": ["还可增加对照实验"],
        "next_step": "补充一组对照样本并记录差异。",
        "confidence": confidence,
        "evidence_refs": refs[:6],
        "ability_applications": [],
    }


def test_validation_accepts_complete_high_confidence_evaluation() -> None:
    case = _case()
    result = validate_evaluation(
        get_task_definition(case.task_id),
        case.answer,
        _raw_evaluation(case),
        valid_evidence_refs=[item["id"] for item in case.evidence_catalog],
    )

    assert result.status == "silver_auto"
    assert result.schema_valid is True
    assert result.reason_codes == ()
    assert result.weight_mismatch_dimensions == ()
    assert result.evaluation is not None
    assert result.evaluation["primary_ability"] == "模型评测"


def test_validation_routes_missing_duplicate_weight_and_invalid_reference() -> None:
    case = _case()
    raw = _raw_evaluation(case)
    raw["dimensions"][-1]["weight"] = 1
    raw["dimensions"].append(dict(raw["dimensions"][0]))
    raw["dimensions"][0]["evidence_refs"] = ["not-in-catalog"]
    del raw["dimensions"][1]

    result = validate_evaluation(
        get_task_definition(case.task_id),
        case.answer,
        raw,
        valid_evidence_refs=[item["id"] for item in case.evidence_catalog],
    )

    assert result.status == "needs_review"
    assert "missing_dimensions" in result.reason_codes
    assert "invalid_evidence_ref" in result.reason_codes
    assert "weight_mismatch" in result.reason_codes


def test_teacher_labeler_uses_cache_and_dry_run_without_gateway_call(tmp_path: Path) -> None:
    case = _case()
    raw = _raw_evaluation(case)

    class FakeGateway:
        calls = 0

        def generate_json(self, system_prompt: str, user_prompt: str, *, model: str):
            self.calls += 1
            return raw

    gateway = FakeGateway()
    settings = Settings(
        dashscope_api_key="test-key",
        trial_teacher_cache_path=str(tmp_path / "teacher.sqlite3"),
    )
    labeler = TeacherLabeler(
        settings=settings,
        cache=TeacherCache(tmp_path / "teacher.sqlite3"),
        gateway=gateway,  # type: ignore[arg-type]
    )

    first = labeler.label(case, model="teacher-v1")
    second = labeler.label(case, model="teacher-v1")
    planned = labeler.label(case, model="teacher-v2", dry_run=True)

    assert first.api_calls == 1
    assert first.cache_hit is False
    assert second.api_calls == 0
    assert second.cache_hit is True
    assert planned.raw is None
    assert planned.api_calls == 0
    assert gateway.calls == 1
    assert labeler.cache.count() == 1


def test_case_generator_validates_quality_level_and_reuses_cache(tmp_path: Path) -> None:
    task = get_task_definition("M-02")
    generated_answer = DynamicTrialAnswer(
        step_answers={step.id: "完成该步骤并记录可复核证据。" for step in task.steps},
        event_decision="调整",
        event_response="补充高风险样本并重跑评测。",
    ).model_dump(mode="json")

    class FakeGateway:
        calls = 0

        def generate_json(self, system_prompt: str, user_prompt: str, *, model: str):
            self.calls += 1
            return generated_answer

    gateway = FakeGateway()
    settings = Settings(
        dashscope_api_key="test-key",
        trial_teacher_cache_path=str(tmp_path / "generation.sqlite3"),
    )
    generator = CaseGenerator(
        settings=settings,
        cache=TeacherCache(tmp_path / "generation.sqlite3"),
        gateway=gateway,  # type: ignore[arg-type]
    )

    first = generator.generate("M-02", quality_level="L3", model="case-v1")
    second = generator.generate("M-02", quality_level="L3", model="case-v1")
    planned = generator.generate("M-02", quality_level="L4", model="case-v1", dry_run=True)
    variant = generator.generate("M-02", quality_level="L3", model="case-v1", variant=2)

    assert first.raw == generated_answer
    assert first.api_calls == 1
    assert second.cache_hit is True
    assert second.api_calls == 0
    assert planned.status == "planned"
    assert planned.raw is None
    assert variant.api_calls == 1
    assert variant.cache_hit is False
    assert gateway.calls == 2

    try:
        generator.generate("M-02", quality_level="L6")
    except ValueError as exc:
        assert "L1-L5" in str(exc)
    else:
        raise AssertionError("非法质量级别未被拒绝")


def test_case_loader_rejects_duplicate_case_ids_and_accepts_chatml(tmp_path: Path) -> None:
    case = _case(case_id="chatml-001")
    payload = {
        "case_id": case.case_id,
        "task_id": case.task_id,
        "messages": [
            {"role": "user", "content": json.dumps({"answer": case.answer.model_dump(mode="json")})}
        ],
    }
    source = tmp_path / "cases.jsonl"
    source.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    loaded = load_case_inputs(source)
    assert loaded[0].case_id == case.case_id

    source.write_text(
        "\n".join(json.dumps(payload, ensure_ascii=False) for _ in range(2)) + "\n",
        encoding="utf-8",
    )
    try:
        load_case_inputs(source)
    except ValueError as exc:
        assert "case_id 重复" in str(exc)
    else:
        raise AssertionError("重复 case_id 未被拒绝")


def test_export_requires_review_for_needs_review_and_never_fabricates_dpo() -> None:
    case = _case()
    base_row = {
        **case.as_record(),
        "evaluation": _raw_evaluation(case),
        "validation": {"status": "needs_review"},
        "teacher": {"model": "teacher-v1", "prompt_version": "v1"},
        "metadata": {"label_status": "needs_review"},
    }
    records, counts = export_sft_records([base_row], prompt_version="v1")
    assert records == []
    assert counts["skipped"] == 1

    reviewed = {**base_row, "metadata": {"human_reviewed": True}}
    records, counts = export_sft_records(
        [reviewed], prompt_version="v1", include_needs_review=True
    )
    assert len(records) == 1
    assert counts["accepted"] == 1

    dpo_records, dpo_counts = export_dpo_records([base_row], prompt_version="v1")
    assert dpo_records == []
    assert dpo_counts["skipped"] == 1

    accepted_row = {
        **base_row,
        "validation": {"status": "silver_auto"},
        "metadata": {"label_status": "silver_auto"},
    }
    records, counts = export_sft_records([accepted_row], prompt_version="v1")
    assert len(records) == 1
    assert counts["accepted"] == 1

    pair = {
        **accepted_row,
        "chosen_evaluation": _raw_evaluation(case),
        "rejected_evaluation": {
            **_raw_evaluation(case),
            "summary": "缺少关键证据。",
            "confidence": "中",
        },
    }
    dpo_records, dpo_counts = export_dpo_records([pair], prompt_version="v1")
    assert len(dpo_records) == 1
    assert dpo_counts["accepted"] == 1
    assert dpo_records[0]["metadata"]["format"] == "dpo-chatml-v1"
