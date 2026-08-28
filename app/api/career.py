import asyncio
import logging
from functools import lru_cache

from fastapi import APIRouter, HTTPException

from app.agents.career_agent import CareerAgent
from app.config import get_settings
from app.knowledge.hybrid import HybridKnowledgeRetriever
from app.knowledge.retriever import KnowledgeRetriever
from app.schemas.career import CareerRecommendation, CareerRecommendationRequest
from app.services.llm_gateway import DashScopeQwenGateway, LLMGatewayError
from app.services.model_response_cache import ModelResponseCache
from app.services.profile_store import ProfileStore
from app.services.task_selector import recommend_trial_task

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/career", tags=["career"])
AI_PRODUCT_MANAGER_DOCUMENT_ID = "job-ai-product-manager-v1"
_recommendation_locks: dict[str, asyncio.Lock] = {}


def _profile_store() -> ProfileStore:
    return ProfileStore(get_settings().profile_db_path)


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


@router.post("/recommendations", response_model=CareerRecommendation)
async def create_career_recommendation(
    request: CareerRecommendationRequest,
) -> CareerRecommendation:
    selected_ids = sorted(dict.fromkeys(request.selected_card_ids))
    cards = _profile_store().get_cards_by_ids(selected_ids)
    if len(cards) != len(selected_ids):
        raise HTTPException(status_code=422, detail="请先选择你已经确认过的能力卡。")

    query = "AI 产品经理 " + " ".join(
        f"{card.title} {card.category} {card.description} {card.detail}" for card in cards
    )
    try:
        retrieved = _knowledge_retriever().search(
            query,
            corpus="career",
            document_id=AI_PRODUCT_MANAGER_DOCUMENT_ID,
            limit=5,
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
                    return CareerRecommendation.model_validate(cached)
                except ValueError:
                    logger.warning("discarding invalid cached career recommendation")

            recommendation = await _career_agent().recommend(
                cards,
                retrieved,
                task_recommendation.selected_task,
                task_recommendation.reason,
            )
            _model_cache().set(
                "career-recommendation",
                cache_key,
                recommendation.model_dump(mode="json"),
            )
            return recommendation
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        logger.exception("career knowledge corpus unavailable")
        raise HTTPException(status_code=503, detail="岗位资料还没有准备好，请先建立本地索引。") from exc
    except LLMGatewayError as exc:
        logger.warning("career recommendation failed reason=%s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        logger.warning("career recommendation invalid reason=%s", exc)
        raise HTTPException(status_code=502, detail="这次建议没有整理成功，请稍后再试。") from exc
