from app.evaluation.multimodal import (
    MultimodalEvaluationCase,
    MultimodalGoldEvidence,
    MultimodalPredictionItem,
    evaluate_multimodal_case,
    render_multimodal_markdown,
    build_multimodal_report,
)


def _case() -> MultimodalEvaluationCase:
    return MultimodalEvaluationCase(
        case_id="mm-001",
        materials=["resume", "portfolio"],
        gold=[
            MultimodalGoldEvidence(
                evidence_id="resume-result",
                material_id="resume",
                page=1,
                bbox=[100, 100, 500, 500],
                quote="负责用户访谈并完成方案迭代",
            ),
            MultimodalGoldEvidence(
                evidence_id="portfolio-impact",
                material_id="portfolio",
                page=2,
                bbox=[200, 200, 700, 700],
                quote="上线后留存率提升 18%",
            ),
        ],
    )


def test_multimodal_case_reports_localization_and_material_coverage() -> None:
    result = evaluate_multimodal_case(
        _case(),
        [
            MultimodalPredictionItem(
                material_id="resume",
                page=1,
                bbox=[120, 120, 480, 480],
                quote="负责用户访谈并完成方案迭代",
            ),
            MultimodalPredictionItem(
                material_id="portfolio",
                page=1,
                bbox=[200, 200, 700, 700],
                quote="上线后留存率提升 18%",
            ),
        ]
    )

    assert result.matched_evidence_count == 1
    assert result.page_hit_rate == 0.5
    assert result.evidence_precision == 0.5
    assert result.evidence_recall == 0.5
    assert result.material_coverage == 0.5
    assert result.localization_iou is not None
    assert result.localization_iou > 0.5


def test_multimodal_report_contains_provenance_and_metrics() -> None:
    case = _case()
    evaluation = evaluate_multimodal_case(case, [])
    report = build_multimodal_report(
        dataset_version="multimodal-v1",
        dataset_sha256="abc123",
        model_id="qwen-vl-ocr",
        cases=[evaluation],
        metadata={"mode": "offline"},
    )

    markdown = render_multimodal_markdown(report)

    assert report.summary.material_coverage == 0.0
    assert report.dataset_sha256 == "abc123"
    assert "定位 IoU" in markdown
    assert "材料覆盖率" in markdown
