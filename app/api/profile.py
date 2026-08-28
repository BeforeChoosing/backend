import asyncio
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.agents.profile_agent import ProfileAgent
from app.config import get_settings
from app.schemas.profile import (
    ConfirmProfileCardsRequest,
    MaterialExtractResponse,
    ProfileExplorationRequest,
    ProfileExplorationResponse,
    ProfileCardPatchRequest,
    ProfileCardsResponse,
    ProfileOverviewResponse,
    ProfileProposalRequest,
    ProfileProposalResponse,
)
from app.schemas.multimodal import MultimodalEvidenceResponse
from app.services.llm_gateway import DashScopeQwenGateway, LLMGatewayError
from app.services.material_extractor import MaterialExtractionError, extract_material_text
from app.services.model_response_cache import ModelResponseCache
from app.services.multimodal_evidence import MultimodalEvidenceExtractor, MultimodalExtractionError
from app.services.profile_store import ProfileStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/profile", tags=["profile"])
MAX_MATERIAL_BYTES = 20 * 1024 * 1024
_exploration_locks: dict[str, asyncio.Lock] = {}
_proposal_locks: dict[str, asyncio.Lock] = {}


def _profile_store() -> ProfileStore:
    return ProfileStore(get_settings().profile_db_path)


def _profile_agent() -> ProfileAgent:
    return ProfileAgent(DashScopeQwenGateway(get_settings()))


def _model_cache() -> ModelResponseCache:
    return ModelResponseCache(_profile_store().db_path)


@router.post("/exploration/messages", response_model=ProfileExplorationResponse)
async def create_profile_exploration_message(
    request: ProfileExplorationRequest,
) -> ProfileExplorationResponse:
    cache_key = ModelResponseCache.fingerprint(
        {
            "prompt_version": ProfileAgent.EXPLORATION_PROMPT_VERSION,
            "model": get_settings().qwen_model,
            "experience_text": request.experience_text,
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "target_role": request.target_role,
            "existing_card_titles": request.existing_card_titles,
        }
    )
    lock = _exploration_locks.setdefault(cache_key, asyncio.Lock())
    async with lock:
        cached = _model_cache().get("profile-exploration", cache_key)
        if cached is not None:
            try:
                return ProfileExplorationResponse.model_validate(cached)
            except ValueError:
                logger.warning("discarding invalid cached profile exploration")
        trace_id = uuid4().hex
        try:
            response = await _profile_agent().explore(request, trace_id)
            _model_cache().set(
                "profile-exploration",
                cache_key,
                response.model_dump(mode="json"),
            )
            return response
        except LLMGatewayError as exc:
            logger.warning("profile exploration failed trace_id=%s reason=%s", trace_id, exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (ValueError, TypeError) as exc:
            logger.warning("profile exploration invalid trace_id=%s reason=%s", trace_id, exc)
            raise HTTPException(status_code=502, detail="本轮补充引导没有整理成功，请稍后重试。") from exc


@router.post("/materials/extract", response_model=MaterialExtractResponse)
async def extract_profile_material(file: UploadFile = File(...)) -> MaterialExtractResponse:
    file_name = Path(file.filename or "upload").name
    data = await file.read(MAX_MATERIAL_BYTES + 1)
    await file.close()
    if len(data) > MAX_MATERIAL_BYTES:
        raise HTTPException(status_code=413, detail="材料文件不能超过 20MB。")
    if not data:
        raise HTTPException(status_code=422, detail="材料文件为空。")
    try:
        text, truncated = extract_material_text(file_name, data)
    except MaterialExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return MaterialExtractResponse(
        file_name=file_name,
        text=text,
        char_count=len(text),
        truncated=truncated,
    )


@router.post(
    "/materials/multimodal-extract",
    response_model=MultimodalEvidenceResponse,
)
async def extract_profile_multimodal_evidence(
    file: UploadFile = File(...),
) -> MultimodalEvidenceResponse:
    """Extract candidate page/region evidence with the configured Qwen-VL model."""

    file_name = Path(file.filename or "upload").name
    data = await file.read(MAX_MATERIAL_BYTES + 1)
    content_type = file.content_type
    await file.close()
    if len(data) > MAX_MATERIAL_BYTES:
        raise HTTPException(status_code=413, detail="材料文件不能超过 20MB。")
    if not data:
        raise HTTPException(status_code=422, detail="材料文件为空。")
    try:
        return await MultimodalEvidenceExtractor().extract(
            file_name=file_name,
            data=data,
            mime_type=content_type,
        )
    except MultimodalExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LLMGatewayError as exc:
        logger.warning("multimodal evidence extraction failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/cards", response_model=ProfileCardsResponse)
def get_profile_cards() -> ProfileCardsResponse:
    return _profile_store().get_profile()


@router.get("/overview", response_model=ProfileOverviewResponse)
def get_profile_overview() -> ProfileOverviewResponse:
    return _profile_store().get_profile_overview()


@router.post("/cards/confirm", response_model=ProfileCardsResponse)
def confirm_profile_cards(request: ConfirmProfileCardsRequest) -> ProfileCardsResponse:
    return _profile_store().confirm_cards(request.cards, request.trace_id)


@router.patch("/cards/{card_id}", response_model=ProfileCardsResponse)
def update_profile_card(
    card_id: str,
    request: ProfileCardPatchRequest,
) -> ProfileCardsResponse:
    try:
        return _profile_store().update_card(card_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="这张能力卡不存在。") from exc


@router.delete("/cards/{card_id}", response_model=ProfileCardsResponse)
def delete_profile_card(card_id: str) -> ProfileCardsResponse:
    try:
        return _profile_store().delete_card(card_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="这张能力卡不存在。") from exc


@router.post("/proposals", response_model=ProfileProposalResponse)
async def create_profile_proposal(
    request: ProfileProposalRequest,
) -> ProfileProposalResponse:
    cache_key = ModelResponseCache.fingerprint(
        {
            "prompt_version": ProfileAgent.PROMPT_VERSION,
            "model": get_settings().qwen_model,
            "request": request.model_dump(mode="json"),
        }
    )
    lock = _proposal_locks.setdefault(cache_key, asyncio.Lock())
    async with lock:
        cached = _model_cache().get("profile-proposal", cache_key)
        if cached is not None:
            try:
                return ProfileProposalResponse.model_validate(cached)
            except ValueError:
                logger.warning("discarding invalid cached profile proposal")
        trace_id = uuid4().hex
        try:
            response = await _profile_agent().propose(request, trace_id)
            _model_cache().set(
                "profile-proposal",
                cache_key,
                response.model_dump(mode="json"),
            )
            return response
        except LLMGatewayError as exc:
            logger.warning("profile proposal failed trace_id=%s reason=%s", trace_id, exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (ValueError, TypeError) as exc:
            logger.warning("profile proposal invalid trace_id=%s reason=%s", trace_id, exc)
            raise HTTPException(status_code=502, detail="模型输出未通过结构化校验，请稍后重试。") from exc
