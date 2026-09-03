"""P0 experiments proving evidence, retrieval and idempotency behavior."""

from __future__ import annotations

import asyncio
import json
import math
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from app.agents.trial_agent import TrialAgent
from app.api import career as career_api
from app.config import Settings, get_settings
from app.evaluation.models import EvaluationCase
from app.evaluation.runner import load_evaluation_cases
from app.knowledge.hybrid import HybridKnowledgeRetriever
from app.knowledge.retriever import KnowledgeChunk, KnowledgeRetriever
from app.knowledge.vector_index import LocalVectorIndex
from app.schemas.career import CareerRecommendation, CareerRecommendationRequest
from app.schemas.profile import CardProposal
from app.schemas.trial import TrialDimensionEvaluation, TrialEvaluation
from app.services.llm_gateway import DashScopeQwenGateway
from app.services.model_response_cache import ModelResponseCache
from app.services.profile_store import ProfileStore
from app.services.trial_scoring import TrialScoringService
from app.services.trial_verification import TrialVerificationService
from app.tasks.catalog import get_task_definition


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _reference_evaluation(case: EvaluationCase) -> tuple[TrialEvaluation, Any, Any]:
    task = get_task_definition(case.task_id)
    bundle = TrialScoringService.build_evidence(task, case.answer, [])
    valid_refs = [item.id for item in bundle.items]
    fallback_ref = valid_refs[0] if valid_refs else "answer:missing"
    dimensions = [
        TrialDimensionEvaluation(
            dimension=criterion.dimension,
            weight=criterion.weight,
            score=case.gold.dimensions.get(criterion.dimension, 60),
            evidence=f"[{fallback_ref}] 该字段包含与评分维度相关的可观察行为。",
            evidence_refs=[fallback_ref],
        )
        for criterion in task.rubric
    ]
    evaluation = TrialEvaluation(
        summary="该作答形成了可按任务 Rubric 核验的结构化结果。",
        dimensions=dimensions,
        primary_ability=task.primary_skill,
        observed_level=case.gold.observed_level,
        level_reason=f"[{fallback_ref}] 等级仅依据本次任务中的可观察证据。",
        process_evidence=["已完成固定任务作答"],
        strengths=["判断包含可追溯依据"],
        gaps=["仍需在其他任务中继续验证"],
        next_step="在下一项固定任务中验证同一能力。",
        confidence="中",
        evidence_refs=[fallback_ref],
    )
    return evaluation, task, bundle


def run_verifier_mutation_experiment(cases_path: str | Path) -> dict[str, Any]:
    """Inject five invalid variants per case and measure deterministic rejection."""

    service = TrialVerificationService(min_evidence_coverage=0.75)
    cases = load_evaluation_cases(cases_path)
    valid_passes = 0
    attacks: list[dict[str, Any]] = []
    for case in cases:
        evaluation, task, bundle = _reference_evaluation(case)
        valid_check = service.check(task, case.answer, bundle, evaluation)
        valid_passes += int(not valid_check.triggered)
        valid_ref = bundle.items[0].id if bundle.items else "answer:missing"

        semantic_mutations: dict[str, TrialEvaluation] = {
            "invalid_evidence_ref": evaluation.model_copy(
                update={
                    "dimensions": [
                        item.model_copy(update={"evidence_refs": ["invented:evidence"]})
                        if index == 0
                        else item
                        for index, item in enumerate(evaluation.dimensions)
                    ]
                }
            ),
            "missing_dimension": evaluation.model_copy(
                update={"dimensions": evaluation.dimensions[1:]}
            ),
            "rubric_mismatch": evaluation.model_copy(
                update={
                    "dimensions": [
                        item.model_copy(update={"weight": max(0, item.weight - 1)})
                        if index == 0
                        else item
                        for index, item in enumerate(evaluation.dimensions)
                    ]
                }
            ),
            "score_without_evidence": evaluation.model_copy(
                update={
                    "dimensions": [
                        item.model_copy(update={"score": 95, "evidence_refs": []})
                        for item in evaluation.dimensions
                    ],
                    "evidence_refs": [valid_ref],
                }
            ),
        }
        for mutation, candidate in semantic_mutations.items():
            check = service.check(task, case.answer, bundle, candidate)
            attacks.append(
                {
                    "case_id": case.case_id,
                    "mutation": mutation,
                    "detected": check.triggered,
                    "stage": "semantic_gate",
                    "reason_codes": check.reason_codes,
                }
            )

        invalid_payload = evaluation.model_dump(mode="json")
        invalid_payload["dimensions"][0]["score"] = 101
        schema_detected = False
        try:
            TrialEvaluation.model_validate(invalid_payload)
        except ValidationError:
            schema_detected = True
        attacks.append(
            {
                "case_id": case.case_id,
                "mutation": "score_out_of_range",
                "detected": schema_detected,
                "stage": "schema_gate",
                "reason_codes": ["schema_validation"] if schema_detected else [],
            }
        )

    attack_count = len(attacks)
    detected_count = sum(int(row["detected"]) for row in attacks)
    return {
        "experiment": "verifier_mutation_v1",
        "case_count": len(cases),
        "mutations_per_case": 5,
        "attack_count": attack_count,
        "detected_count": detected_count,
        "attack_detection_rate": detected_count / attack_count if attack_count else 0.0,
        "valid_case_pass_rate": valid_passes / len(cases) if cases else 0.0,
        "false_rejection_rate": 1 - valid_passes / len(cases) if cases else 0.0,
        "api_calls": 0,
        "details": attacks,
    }


def _rank_metrics(ranks: list[int | None], *, k_values: tuple[int, ...] = (1, 3, 5)) -> dict[str, float]:
    total = max(1, len(ranks))
    metrics = {
        f"hit_at_{k}": sum(rank is not None and rank <= k for rank in ranks) / total
        for k in k_values
    }
    metrics["mrr_at_5"] = sum(1 / rank for rank in ranks if rank is not None and rank <= 5) / total
    metrics["ndcg_at_5"] = sum(
        1 / math.log2(rank + 1) for rank in ranks if rank is not None and rank <= 5
    ) / total
    return metrics


class _NoRemoteGateway:
    def embed(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("实验要求使用已有 Embedding 缓存")

    def rerank(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("实验要求使用已有 Rerank 缓存")


def run_rag_ablation_experiment(
    cases_path: str | Path,
    *,
    settings: Settings | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Evaluate FTS, vector, fusion and fusion+rerank using the existing index/cache."""

    settings = settings or get_settings()
    raw_cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    base = KnowledgeRetriever(settings.knowledge_dir, settings.knowledge_db_path)
    blocker = _NoRemoteGateway()
    # The ablation's fourth arm must exercise fusion + rerank explicitly;
    # production defaults may select the better semantic-vector strategy.
    hybrid_settings = replace(settings, rag_retriever_mode="hybrid")
    hybrid = HybridKnowledgeRetriever(
        settings.knowledge_dir,
        settings.knowledge_db_path,
        settings=hybrid_settings,
        retriever=base,
        embedding_gateway=blocker,
        rerank_gateway=blocker,
        cache=ModelResponseCache(base.db_path),
    )
    index = LocalVectorIndex(base.db_path)
    if not index.ready:
        raise ValueError("本地向量索引为空")

    arm_ranks: dict[str, list[int | None]] = {
        "fts5": [],
        "vector": [],
        "fts_vector_fusion": [],
        "fts_vector_rerank": [],
    }
    details: list[dict[str, Any]] = []
    candidate_limit = max(hybrid.candidate_limit, limit)
    for row in raw_cases:
        query = str(row["query"])
        expected = str(row["expected_heading"]).lower()
        lexical = base.search(
            query,
            corpus="career",
            document_id="job-ai-product-manager-v1",
            limit=candidate_limit,
        )
        vector = hybrid._query_vector(query)
        vector_hits = index.search(
            vector,
            corpus="career",
            document_id="job-ai-product-manager-v1",
            model=settings.bailian_embedding_model,
            dimension=settings.bailian_embedding_dimension,
            source_fingerprint=base.source_fingerprint,
            limit=candidate_limit,
        )
        vector_chunks = base.get_chunks_by_ids([hit.chunk_id for hit in vector_hits])
        vector_by_chunk = {chunk.id: chunk for chunk in vector_chunks}
        vector_ranked = [
            replace(vector_by_chunk[hit.chunk_id], score=hit.score)
            for hit in vector_hits
            if hit.chunk_id in vector_by_chunk
        ]
        candidate_ids = list(dict.fromkeys(
            [chunk.id for chunk in lexical] + [hit.chunk_id for hit in vector_hits]
        ))[:candidate_limit]
        candidates = base.get_chunks_by_ids(candidate_ids)
        fusion = hybrid._combined_fallback(
            candidates,
            {chunk.id: chunk for chunk in lexical},
            {hit.chunk_id: hit.score for hit in vector_hits},
        )
        reranked = hybrid.search(
            query,
            corpus="career",
            document_id="job-ai-product-manager-v1",
            limit=limit,
        )
        if not hybrid.last_diagnostics.get("rerank_used"):
            raise ValueError(f"{query} 未命中已有 Rerank 缓存")
        results = {
            "fts5": lexical[:limit],
            "vector": vector_ranked[:limit],
            "fts_vector_fusion": fusion[:limit],
            "fts_vector_rerank": reranked[:limit],
        }
        row_detail: dict[str, Any] = {"query": query, "expected_heading": row["expected_heading"]}
        for arm, chunks in results.items():
            rank = next(
                (
                    index_ + 1
                    for index_, chunk in enumerate(chunks)
                    if expected in " > ".join(chunk.heading_path).lower()
                ),
                None,
            )
            arm_ranks[arm].append(rank)
            row_detail[arm] = rank
        details.append(row_detail)
    return {
        "experiment": "rag_ablation_v1",
        "query_count": len(raw_cases),
        "k": limit,
        "arms": {arm: _rank_metrics(ranks) for arm, ranks in arm_ranks.items()},
        "embedding_model": settings.bailian_embedding_model,
        "rerank_model": settings.bailian_rerank_model,
        "api_calls": 0,
        "cache_policy": "existing_query_embedding_and_rerank_cache_only",
        "details": details,
    }


class _FixedRetriever:
    def search(self, *args: Any, **kwargs: Any) -> list[KnowledgeChunk]:
        return [
            KnowledgeChunk(
                id="experiment-career-chunk",
                document_id="job-ai-product-manager-v1",
                document_title="AI 产品经理岗位知识库",
                corpus="career",
                content="AI 产品经理需要将用户问题转化为可验证方案。",
                heading_path=("AI 产品经理", "用户研究"),
                source_locator="jobs/ai_product_manager.md#用户研究",
                trust_level="secondary_summary",
                source_note="实验固定资料",
                score=1.0,
            )
        ]


class _CountingCareerAgent:
    def __init__(self, delay_seconds: float = 0.03):
        self.calls = 0
        self.delay_seconds = delay_seconds

    async def recommend(self, cards: Any, retrieved: Any, next_task: Any, next_task_reason: str) -> CareerRecommendation:
        self.calls += 1
        await asyncio.sleep(self.delay_seconds)
        return CareerRecommendation(
            summary="已形成带来源的职业探索建议。",
            supported=[
                {
                    "claim": "用户研究能力可支持需求分析。",
                    "card_ids": [cards[0].id],
                    "citation_ids": [retrieved[0].id],
                }
            ],
            unknowns=["仍需通过固定任务验证复杂场景判断。"],
            next_task_id=next_task.id,
            next_task_title=next_task.title,
            next_task_reason=next_task_reason,
            confidence="中",
            citations=[
                {
                    "id": retrieved[0].id,
                    "document_title": retrieved[0].document_title,
                    "source_locator": retrieved[0].source_locator,
                    "content": retrieved[0].content,
                    "trust_level": retrieved[0].trust_level,
                    "source_note": retrieved[0].source_note,
                }
            ],
        )


def _experiment_card() -> CardProposal:
    return CardProposal(
        id="experiment-card",
        title="用户研究",
        category="洞察分析",
        description="根据用户反馈识别可行动问题。",
        detail="通过访谈、归类和复核形成需求假设。",
        evidence_quote="整理用户反馈并调整方案。",
        source_refs=["input:experience_text"],
        next_verification="在固定任务中验证分析过程。",
        match_reason="经历中包含反馈整理与方案调整。",
        workplace_application="支持 AI 产品需求分析。",
    )


async def run_idempotency_experiment(request_count: int = 20) -> dict[str, Any]:
    """Compare naive calls with the production recommendation lock and cache."""

    with tempfile.TemporaryDirectory(prefix="before-choosing-idempotency-") as directory:
        store = ProfileStore(Path(directory) / "profile.db")
        store.confirm_cards([_experiment_card()], trace_id="p0-idempotency")
        request = CareerRecommendationRequest(selected_card_ids=["experiment-card"])
        agent = _CountingCareerAgent()
        original_profile_store = career_api._profile_store
        original_retriever = career_api._knowledge_retriever
        original_agent = career_api._career_agent
        career_api._recommendation_locks.clear()
        try:
            career_api._profile_store = lambda: store
            career_api._knowledge_retriever = lambda: _FixedRetriever()
            career_api._career_agent = lambda: agent
            started = time.perf_counter()
            responses = await asyncio.gather(
                *[career_api.create_career_recommendation(request) for _ in range(request_count)]
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
        finally:
            career_api._profile_store = original_profile_store
            career_api._knowledge_retriever = original_retriever
            career_api._career_agent = original_agent
            career_api._recommendation_locks.clear()
        payloads = [response.model_dump(mode="json") for response in responses]
        unique_results = len({json.dumps(payload, sort_keys=True, ensure_ascii=False) for payload in payloads})
        actual_calls = agent.calls
        return {
            "experiment": "idempotency_concurrency_v1",
            "request_count": request_count,
            "naive_model_calls": request_count,
            "actual_model_calls": actual_calls,
            "avoided_model_calls": request_count - actual_calls,
            "call_reduction_rate": (request_count - actual_calls) / request_count,
            "cache_or_lock_reuse_rate": (request_count - actual_calls) / request_count,
            "unique_result_count": unique_results,
            "result_consistency_rate": 1.0 if unique_results == 1 else 0.0,
            "elapsed_ms": elapsed_ms,
            "paid_api_calls": 0,
            "method": "production career endpoint function with real fingerprint cache and asyncio lock; model replaced by counting fixture",
        }


def _evidence_stripped_case(case: EvaluationCase) -> EvaluationCase:
    answer = case.answer.model_copy(
        deep=True,
        update={
            "step_answers": {
                step_id: "已综合现有信息完成分析，并形成后续可执行的优化建议。"
                for step_id in case.answer.step_answers
            },
            "viewed_material_ids": [],
            "evidence_refs": [],
            "step_revisions": {},
            "card_play_rationale": "将已有能力用于分析问题并提出方案。",
            "validation_hypothesis": "方案实施后应当获得更好的结果。",
            "event_response": "根据新情况调整方案，并继续观察结果。",
        },
    )
    return case.model_copy(update={"answer": answer})


def _weighted_score(evaluation: TrialEvaluation) -> float:
    total_weight = sum(item.weight for item in evaluation.dimensions)
    if total_weight <= 0:
        return _mean(float(item.score) for item in evaluation.dimensions)
    return sum(item.score * item.weight for item in evaluation.dimensions) / total_weight


def _level_value(level: str) -> int:
    return {"证据不足": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}.get(level, 0)


async def run_evidence_sensitivity_experiment(
    cases_path: str | Path,
    *,
    output_dir: str | Path,
    settings: Settings | None = None,
    concurrency: int = 3,
) -> dict[str, Any]:
    """Evaluate 12 original/stripped pairs with exactly 24 meaningful Qwen calls."""

    settings = settings or get_settings()
    if not settings.qwen_configured:
        raise ValueError("未配置 DASHSCOPE_API_KEY")
    settings = replace(settings, request_timeout_seconds=max(settings.request_timeout_seconds, 120.0))
    all_cases = load_evaluation_cases(cases_path)
    selected: list[EvaluationCase] = []
    seen_tasks: set[str] = set()
    for case in all_cases:
        if case.task_id not in seen_tasks:
            selected.append(case)
            seen_tasks.add(case.task_id)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache = ModelResponseCache(output / "cache.sqlite3")
    gateway = DashScopeQwenGateway(settings)
    agent = TrialAgent(gateway, prompt_variant="prompt", model_override=settings.qwen_model)
    verifier = TrialVerificationService(min_evidence_coverage=0.75)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    api_calls = 0
    api_calls_lock = asyncio.Lock()

    async def evaluate(case: EvaluationCase, variant: str) -> dict[str, Any]:
        nonlocal api_calls
        task = get_task_definition(case.task_id)
        bundle = TrialScoringService.build_evidence(task, case.answer, [])
        key = ModelResponseCache.fingerprint(
            {
                "experiment": "evidence-sensitivity-v1",
                "model": settings.qwen_model,
                "prompt_version": TrialAgent.PROMPT_VERSION,
                "case_id": case.case_id,
                "variant": variant,
                "answer": case.answer.model_dump(mode="json"),
            }
        )
        cached = cache.get("evidence-sensitivity", key)
        if cached is not None:
            evaluation = TrialEvaluation.model_validate(cached)
            cache_hit = True
        else:
            async with semaphore:
                evaluation = await agent.evaluate_dynamic(task, case.answer, [], bundle)
            evaluation, bundle = TrialScoringService.finalize_dynamic(
                task, case.answer, [], evaluation
            )
            cache.set("evidence-sensitivity", key, evaluation.model_dump(mode="json"))
            async with api_calls_lock:
                api_calls += 1
            cache_hit = False
        check = verifier.check(task, case.answer, bundle, evaluation)
        return {
            "variant": variant,
            "weighted_score": round(_weighted_score(evaluation), 4),
            "observed_level": evaluation.observed_level,
            "verification_triggered": check.triggered,
            "verification_reason_codes": check.reason_codes,
            "evidence_coverage": check.evidence_coverage,
            "cache_hit": cache_hit,
            "evaluation": evaluation.model_dump(mode="json"),
        }

    async def run_pair(case: EvaluationCase) -> dict[str, Any]:
        stripped = _evidence_stripped_case(case)
        original, degraded = await asyncio.gather(
            evaluate(case, "original"),
            evaluate(stripped, "evidence_stripped"),
        )
        return {
            "case_id": case.case_id,
            "task_id": case.task_id,
            "original": original,
            "evidence_stripped": degraded,
            "score_delta": round(original["weighted_score"] - degraded["weighted_score"], 4),
            "level_delta": _level_value(original["observed_level"]) - _level_value(degraded["observed_level"]),
        }

    pairs = await asyncio.gather(*[run_pair(case) for case in selected])
    report = {
        "experiment": "trial_evidence_sensitivity_v1",
        "model": settings.qwen_model,
        "prompt_version": TrialAgent.PROMPT_VERSION,
        "pair_count": len(pairs),
        "case_count": len(pairs) * 2,
        "successful_result_count": len(pairs) * 2,
        "unique_successful_requests": len(pairs) * 2,
        "api_calls_this_run": api_calls,
        "cache_hits_this_run": len(pairs) * 2 - api_calls,
        "mean_original_score": _mean(
            float(row["original"]["weighted_score"]) for row in pairs
        ),
        "mean_stripped_score": _mean(
            float(row["evidence_stripped"]["weighted_score"]) for row in pairs
        ),
        "score_drop_rate": _mean(float(row["score_delta"] > 0) for row in pairs),
        "mean_score_delta": _mean(float(row["score_delta"]) for row in pairs),
        "level_drop_rate": _mean(float(row["level_delta"] > 0) for row in pairs),
        "unsupported_high_score_rate": _mean(
            float(row["evidence_stripped"]["weighted_score"] >= 70) for row in pairs
        ),
        "stripped_verifier_trigger_rate": _mean(
            float(row["evidence_stripped"]["verification_triggered"]) for row in pairs
        ),
        "pairs": pairs,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
