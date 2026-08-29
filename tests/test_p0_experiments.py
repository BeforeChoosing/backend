import asyncio

from app.evaluation.p0_experiments import (
    _rank_metrics,
    run_idempotency_experiment,
    run_verifier_mutation_experiment,
)


def test_verifier_mutations_are_detected_without_rejecting_reference_cases():
    report = run_verifier_mutation_experiment(
        "datasets/trial_agent/eval/locked_cases.v1.jsonl"
    )

    assert report["case_count"] == 24
    assert report["attack_count"] == 120
    assert report["attack_detection_rate"] == 1.0
    assert report["valid_case_pass_rate"] == 1.0
    assert report["api_calls"] == 0


def test_rank_metrics_include_hit_mrr_and_ndcg():
    metrics = _rank_metrics([1, 2, None, 5])

    assert metrics["hit_at_1"] == 0.25
    assert metrics["hit_at_3"] == 0.5
    assert metrics["hit_at_5"] == 0.75
    assert metrics["mrr_at_5"] == (1 + 0.5 + 0.2) / 4
    assert 0 < metrics["ndcg_at_5"] < 1


def test_production_cache_and_lock_collapse_concurrent_requests():
    report = asyncio.run(run_idempotency_experiment(20))

    assert report["request_count"] == 20
    assert report["actual_model_calls"] == 1
    assert report["avoided_model_calls"] == 19
    assert report["result_consistency_rate"] == 1.0
    assert report["paid_api_calls"] == 0
