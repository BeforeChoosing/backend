from app.evaluation.experiment_evidence import (
    build_experiment_evidence_report,
    render_experiment_evidence_markdown,
)


def test_build_experiment_evidence_report_compares_all_three_pipelines() -> None:
    report = build_experiment_evidence_report(
        trial={
            "model_id": "qwen-plus",
            "dataset_version": "locked-v1",
            "dataset_sha256": "abc",
            "arms": [
                {
                    "arm": "base_qwen",
                    "case_count": 24,
                    "valid_schema_rate": 0.0,
                    "mean_api_calls": 1.0,
                    "mean_latency_ms": 50_000,
                },
                {
                    "arm": "prompt_hardened",
                    "case_count": 24,
                    "valid_schema_rate": 1.0,
                    "dimension_score_mae": 8.75,
                    "level_exact_rate": 0.75,
                    "level_within_one_rate": 0.95,
                    "evidence_precision": 1.0,
                    "evidence_recall": 1.0,
                    "invalid_evidence_ref_rate": 0.0,
                    "mean_api_calls": 1.0,
                    "mean_latency_ms": 33_000,
                },
            ],
        },
        rag={
            "before": {"hit_at_k": 0.25, "mrr_at_k": 0.25, "details": ["a", "b"]},
            "after": {"hit_at_k": 0.875, "mrr_at_k": 0.875, "model": "qwen3-rerank"},
            "api_calls_estimate": 4,
        },
        multimodal=[
            {
                "model": "qwen-vl-ocr",
                "source_file": "sample.pdf",
                "page_count": 8,
                "mean_character_similarity": 0.45,
                "empty_prediction_rate": 0.0,
                "mean_latency_ms": 9_400,
            },
            {
                "model": "qwen3-vl-plus",
                "source_file": "sample.pdf",
                "page_count": 8,
                "mean_character_similarity": 0.99,
                "empty_prediction_rate": 0.0,
                "mean_latency_ms": 10_100,
            },
        ],
    )

    assert report["trial_agent"]["schema_valid_rate_delta"] == 1.0
    assert report["rag"]["hit_at_k_delta"] == 0.625
    assert report["multimodal_ocr"]["best_model"] == "qwen3-vl-plus"
    assert report["multimodal_ocr"]["similarity_delta_vs_baseline"] == 0.54
    assert report["clean_run_api_calls"] == 68
    markdown = render_experiment_evidence_markdown(report)
    assert "Schema 合法率：0.0% → 100.0%" in markdown
    assert "qwen3-vl-plus" in markdown
