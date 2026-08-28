from app.evaluation.metrics import evaluate_case
from app.evaluation.models import EvaluationCase, GoldTrialEvaluation
from app.evaluation.report import build_report, render_markdown
from app.schemas.task_catalog import DynamicTrialAnswer
from app.schemas.trial import TrialDimensionEvaluation, TrialEvaluation


def _case() -> EvaluationCase:
    return EvaluationCase(
        case_id="m02-contract-001",
        task_id="M-02",
        answer=DynamicTrialAnswer(
            step_answers={
                "scenarios": "覆盖正常、边界和高风险场景。",
                "failures": "按意图错误、风格错误和安全错误分类。",
                "budget": "按频率和失败成本分配人工样本。",
                "threshold": "严重语气错误率超过门槛则回退。",
                "event": "提高职场委婉表达样本权重并重新评审。",
            },
            event_decision="调整",
            event_response="把严重语气错误率设为上线门槛。",
        ),
        gold=GoldTrialEvaluation(
            dimensions={"模型评测": 80, "用户洞察": 70},
            observed_level="L3",
            evidence_refs=["answer:scenarios", "answer:event"],
        ),
    )


def test_evaluate_case_reports_score_error_level_and_grounded_refs() -> None:
    case = _case()
    prediction = TrialEvaluation(
        summary="覆盖关键场景并设置门槛。",
        dimensions=[
            TrialDimensionEvaluation(
                dimension="模型评测",
                score=75,
                evidence="覆盖高风险样本。",
                evidence_refs=["answer:scenarios", "invented:ref"],
            ),
            TrialDimensionEvaluation(
                dimension="用户洞察",
                score=50,
                evidence="按场景分组。",
                evidence_refs=["answer:failures"],
            ),
        ],
        observed_level="L3",
        confidence="中",
        strengths=["有场景划分"],
        gaps=["需要更清楚的通过标准"],
        next_step="补充评审一致性规则。",
    )

    result = evaluate_case(case, prediction, valid_evidence_refs={
        "answer:scenarios",
        "answer:failures",
        "answer:event",
    })

    assert result.schema_valid is True
    assert result.dimension_score_mae == 12.5
    assert result.level_exact_match is True
    assert result.evidence_precision == 2 / 3
    assert result.evidence_recall == 0.5
    assert result.invalid_evidence_ref_count == 1


def test_evaluate_case_marks_invalid_prediction_without_raising() -> None:
    case = _case()

    result = evaluate_case(case, {"not": "a trial evaluation"}, valid_evidence_refs=set())

    assert result.schema_valid is False
    assert result.error_code == "invalid_schema"


def test_report_contains_arm_comparison_and_provenance() -> None:
    case = _case()
    result = evaluate_case(
        case,
        {
            "summary": "完成评价。",
            "dimensions": [],
            "confidence": "低",
            "strengths": [],
            "gaps": [],
            "next_step": "补充证据。",
        },
        valid_evidence_refs=set(),
        api_calls=1,
    )

    report = build_report(
        dataset_version="trial-agent-v1",
        dataset_sha256="abc123",
        model_id="qwen-plus",
        prompt_version="trial-v3-evidence",
        cases_by_arm={"base_qwen": [result]},
        metadata={"git_commit": "deadbeef"},
    )
    markdown = render_markdown(report)

    assert report.arms[0].arm == "base_qwen"
    assert report.arms[0].valid_schema_rate == 1.0
    assert report.metadata["git_commit"] == "deadbeef"
    assert "base_qwen" in markdown
    assert "dataset_sha256" in markdown
