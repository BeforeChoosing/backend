import asyncio
import logging
import time
from functools import lru_cache

from fastapi import APIRouter, HTTPException

from app.agents.career_agent import CareerAgent
from app.config import get_settings
from app.knowledge.hybrid import HybridKnowledgeRetriever
from app.knowledge.query_planner import QUERY_PLAN_VERSION, build_career_queries
from app.knowledge.retriever import KnowledgeRetriever
from app.schemas.career import (
    CareerRecommendation,
    CareerRecommendationRequest,
    CareerSupport,
)
from app.services.llm_gateway import DashScopeQwenGateway, LLMGatewayError, llm_error_status
from app.services.runtime_log import log_event
from app.services.audit_log import record_business_event
from app.services.model_response_cache import ModelResponseCache
from app.services.profile_store import ProfileStore
from app.services.user_data import user_data_path
from app.services.task_selector import recommend_trial_task

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/career", tags=["career"])
AI_PRODUCT_MANAGER_DOCUMENT_ID = "job-ai-product-manager-v1"
_recommendation_locks: dict[str, asyncio.Lock] = {}


def _profile_store() -> ProfileStore:
    return ProfileStore(user_data_path(get_settings().profile_db_path))


@lru_cache(maxsize=1)
def _knowledge_retriever() -> KnowledgeRetriever | HybridKnowledgeRetriever:
    settings = get_settings()
    if settings.rag_retriever_mode.lower() == "fts":
        return KnowledgeRetriever(settings.knowledge_dir, settings.knowledge_db_path)
    return HybridKnowledgeRetriever(
        settings.knowledge_dir,
        settings.knowledge_db_path,
        settings=settings,
    )


def _career_agent() -> CareerAgent:
    return CareerAgent(DashScopeQwenGateway(get_settings()))


def _model_cache() -> ModelResponseCache:
    return ModelResponseCache(_profile_store().db_path)


def _record_recommendation_event(
    request: CareerRecommendationRequest,
    recommendation: CareerRecommendation,
    *,
    cached: bool,
) -> None:
    try:
        record_business_event(
            get_settings().profile_db_path,
            event_type="career_recommendation",
            action="career.recommendation.generate",
            metadata={
                "selected_card_ids": sorted(dict.fromkeys(request.selected_card_ids)),
                "target_role": request.target_role,
                "next_task_id": recommendation.next_task_id,
                "confidence": recommendation.confidence,
                "citation_count": len(recommendation.citations),
                "cached": cached,
            },
        )
    except Exception:  # noqa: BLE001 - audit must not block career exploration
        logger.exception("failed to record career recommendation event")


def _apply_retrieval_coverage_guard(
    recommendation: CareerRecommendation,
    cards: list,
    retrieved: list,
    retriever: object,
) -> CareerRecommendation:
    """Prevent a recommendation from citing cards absent from retrieved support."""

    diagnostics = getattr(retriever, "last_diagnostics", {})
    per_query_ids = diagnostics.get("per_query_result_ids")
    if not isinstance(per_query_ids, list) or len(per_query_ids) < len(cards) + 1:
        # FTS-only or test doubles do not expose per-card retrieval evidence;
        # retain the existing behavior rather than guessing coverage.
        return recommendation

    retrieved_ids = {chunk.id for chunk in retrieved}
    covered_ids: set[str] = set()
    missing_titles: list[str] = []
    for index, card in enumerate(cards, start=1):
        candidate_ids = per_query_ids[index]
        candidate_ids = candidate_ids if isinstance(candidate_ids, list) else []
        if retrieved_ids.intersection(str(item) for item in candidate_ids):
            covered_ids.add(card.id)
        else:
            missing_titles.append(card.title)

    if not missing_titles:
        return recommendation

    supports: list[CareerSupport] = []
    for support in recommendation.supported:
        valid_card_ids = [card_id for card_id in support.card_ids if card_id in covered_ids]
        if not valid_card_ids:
            continue
        supports.append(
            support.model_copy(update={"card_ids": valid_card_ids})
        )

    unknowns = list(recommendation.unknowns)
    unknowns.insert(
        0,
        "岗位资料暂未覆盖能力卡“{}”，本次不据此作支持性结论。".format(
            "、".join(missing_titles)
        ),
    )
    unknowns = unknowns[:6]
    confidence = recommendation.confidence
    if confidence == "高":
        confidence = "中"
    return recommendation.model_copy(
        update={
            "supported": supports,
            "unknowns": unknowns,
            "confidence": confidence,
        }
    )


@router.post("/recommendations", response_model=CareerRecommendation)
async def create_career_recommendation(
    request: CareerRecommendationRequest,
) -> CareerRecommendation:
    selected_ids = sorted(dict.fromkeys(request.selected_card_ids))
    cards = _profile_store().get_cards_by_ids(selected_ids)
    if len(cards) != len(selected_ids):
        raise HTTPException(status_code=422, detail="请先选择你已经确认过的能力卡。")

    try:
        retriever = _knowledge_retriever()
        planned_queries = build_career_queries(cards, target_role=request.target_role)
        retrieval_started = time.perf_counter()
        try:
            if hasattr(retriever, "search_many"):
                retrieved = retriever.search_many(
                    planned_queries,
                    corpus="career",
                    document_id=AI_PRODUCT_MANAGER_DOCUMENT_ID,
                    limit=5,
                )
            else:
                # Keep an explicit FTS fallback for older/local retriever instances.
                query = " ".join(planned_queries)
                retrieved = retriever.search(
                    query,
                    corpus="career",
                    document_id=AI_PRODUCT_MANAGER_DOCUMENT_ID,
                    limit=5,
                )
        except Exception as exc:  # noqa: BLE001 - preserve the API error mapping
            log_event(
                "knowledge_retrieval_failed",
                level="error",
                error=exc,
                stage="career",
                corpus="career",
                query_count=len(planned_queries),
                retriever_mode=str(
                    getattr(retriever, "last_diagnostics", {}).get("mode", "unknown")
                )[:80],
                duration_ms=round((time.perf_counter() - retrieval_started) * 1000, 3),
            )
            raise
        diagnostics = getattr(retriever, "last_diagnostics", {})
        per_query_ids = diagnostics.get("per_query_result_ids")
        candidate_count = (
            len({str(item) for values in per_query_ids if isinstance(values, list) for item in values})
            if isinstance(per_query_ids, list)
            else None
        )
        log_event(
            "knowledge_retrieval_completed",
            level="info",
            stage="career",
            corpus="career",
            query_count=len(planned_queries),
            candidate_count=candidate_count,
            hit_count=len(retrieved),
            document_count=len({chunk.document_id for chunk in retrieved}),
            retriever_mode=str(diagnostics.get("mode", "unknown"))[:80],
            vector_used=bool(diagnostics.get("vector_used", False)),
            rerank_used=bool(diagnostics.get("rerank_used", False)),
            adaptive_rerank_triggered=bool(
                diagnostics.get("adaptive_rerank_triggered", False)
            ),
            embedding_batch_calls=diagnostics.get("embedding_batch_calls"),
            query_coverage=diagnostics.get("query_coverage"),
            fallback=not bool(diagnostics.get("vector_used", False)),
            duration_ms=round((time.perf_counter() - retrieval_started) * 1000, 3),
        )
        if not retrieved:
            raise HTTPException(status_code=503, detail="暂时没有找到可参考的岗位资料。")
        task_recommendation = recommend_trial_task(
            cards,
            _profile_store().get_completed_task_ids(),
            evidence_records=_profile_store().get_evidence_records(),
            target_role=request.target_role,
        )
        cache_key = ModelResponseCache.fingerprint(
            {
                "prompt_version": CareerAgent.PROMPT_VERSION,
                "rag_query_plan_version": QUERY_PLAN_VERSION,
                "model": get_settings().qwen_model,
                "cards": [card.model_dump(mode="json") for card in cards],
                "retrieved": [chunk.__dict__ for chunk in retrieved],
                "next_task": task_recommendation.selected_task.model_dump(mode="json"),
                "next_task_reason": task_recommendation.reason,
            }
        )
        lock = _recommendation_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            cached = _model_cache().get("career-recommendation", cache_key)
            if cached is not None:
                try:
                    guarded = _apply_retrieval_coverage_guard(
                        CareerRecommendation.model_validate(cached),
                        cards,
                        retrieved,
                        retriever,
                    )
                    _record_recommendation_event(request, guarded, cached=True)
                    return guarded
                except ValueError:
                    logger.warning("discarding invalid cached career recommendation")

            recommendation = await _career_agent().recommend(
                cards,
                retrieved,
                task_recommendation.selected_task,
                task_recommendation.reason,
            )
            recommendation = _apply_retrieval_coverage_guard(
                recommendation,
                cards,
                retrieved,
                retriever,
            )
            _model_cache().set(
                "career-recommendation",
                cache_key,
                recommendation.model_dump(mode="json"),
            )
            _record_recommendation_event(request, recommendation, cached=False)
            return recommendation
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        logger.exception("career knowledge corpus unavailable")
        raise HTTPException(status_code=503, detail="岗位资料还没有准备好，请先建立本地索引。") from exc
    except LLMGatewayError as exc:
        logger.warning("career recommendation failed reason=%s", exc)
        raise HTTPException(status_code=llm_error_status(exc), detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        logger.warning("career recommendation invalid reason=%s", exc)
        raise HTTPException(status_code=502, detail="这次建议没有整理成功，请稍后再试。") from exc
