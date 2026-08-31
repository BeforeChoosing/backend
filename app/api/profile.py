import asyncio
import hashlib
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

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
from app.services.llm_gateway import DashScopeQwenGateway, LLMGatewayError, llm_error_status
from app.services.material_extractor import MaterialExtractionError, extract_material_text
from app.services.model_response_cache import ModelResponseCache
from app.services.multimodal_evidence import MultimodalEvidenceExtractor, MultimodalExtractionError
from app.services.audit_log import record_business_event
from app.services.conversation_store import ConversationStore
from app.services.request_context import get_request_context
from app.services.profile_exploration_controller import (
    CONTROLLER_VERSION,
    apply_exploration_controller,
)
from app.services.profile_store import ProfileStore
from app.services.user_data import user_data_path

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/profile", tags=["profile"])
MAX_MATERIAL_BYTES = 20 * 1024 * 1024
_exploration_locks: dict[str, asyncio.Lock] = {}
_proposal_locks: dict[str, asyncio.Lock] = {}


def _profile_store() -> ProfileStore:
    return ProfileStore(user_data_path(get_settings().profile_db_path))


def _profile_agent() -> ProfileAgent:
    return ProfileAgent(DashScopeQwenGateway(get_settings()))


def _model_cache() -> ModelResponseCache:
    return ModelResponseCache(_profile_store().db_path)


def _record_business_event(
    event_type: str,
    action: str,
    metadata: dict[str, object] | None = None,
) -> None:
    try:
        record_business_event(
            get_settings().profile_db_path,
            event_type=event_type,
            action=action,
            metadata=metadata,
        )
    except Exception:  # noqa: BLE001 - audit must never block a user operation
        logger.exception("failed to record profile business event action=%s", action)


def _record_exploration_turn(
    request: ProfileExplorationRequest,
    response: ProfileExplorationResponse,
) -> None:
    context = get_request_context()
    if context.app_mode != "use" or not context.user_id:
        return
    try:
        ConversationStore(get_settings().profile_db_path).record_turn(
            user_id=context.user_id,
            request_id=request.request_id or context.request_id,
            trace_id=response.trace_id,
            experience_text=request.experience_text,
            messages=[message.model_dump(mode="json") for message in request.messages],
            response=response.model_dump(mode="json"),
        )
    except Exception:  # noqa: BLE001 - conversation persistence must not block a response
        logger.exception("failed to persist profile exploration turn")
    _record_business_event(
        "profile_chat",
        "profile.exploration.message",
        {
            "trace_id": response.trace_id,
            "message_count": len(request.messages),
            "experience_chars": len(request.experience_text),
            "focus_dimension": response.focus_dimension,
            "ready_for_proposal": response.ready_for_proposal,
        },
    )


def _material_metadata(file_name: str, data: bytes, *, content_type: str | None) -> dict[str, object]:
    return {
        "file_name": file_name,
        "mime_type": content_type or "application/octet-stream",
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


@router.post("/exploration/messages", response_model=ProfileExplorationResponse)
async def create_profile_exploration_message(
    request: ProfileExplorationRequest,
) -> ProfileExplorationResponse:
    cache_key = ModelResponseCache.fingerprint(
        {
            "prompt_version": ProfileAgent.EXPLORATION_PROMPT_VERSION,
            "controller_version": CONTROLLER_VERSION,
            "model": get_settings().qwen_model,
            "experience_text": request.experience_text,
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "focus_history": request.focus_history,
            "target_role": request.target_role,
            "existing_card_titles": request.existing_card_titles,
        }
    )
    lock = _exploration_locks.setdefault(cache_key, asyncio.Lock())
    async with lock:
        cached = _model_cache().get("profile-exploration", cache_key)
        if cached is not None:
            try:
                response = ProfileExplorationResponse.model_validate(cached)
                _record_exploration_turn(request, response)
                return response
            except ValueError:
                logger.warning("discarding invalid cached profile exploration")
        trace_id = uuid4().hex
        try:
            response = apply_exploration_controller(
                await _profile_agent().explore(request, trace_id),
                request,
            )
            _model_cache().set(
                "profile-exploration",
                cache_key,
                response.model_dump(mode="json"),
            )
            _record_exploration_turn(request, response)
            return response
        except LLMGatewayError as exc:
            logger.warning("profile exploration failed trace_id=%s reason=%s", trace_id, exc)
            raise HTTPException(status_code=llm_error_status(exc), detail=str(exc)) from exc
        except (ValueError, TypeError) as exc:
            logger.warning("profile exploration invalid trace_id=%s reason=%s", trace_id, exc)
            raise HTTPException(status_code=502, detail="本轮补充引导没有整理成功，请稍后重试。") from exc


@router.post("/materials/extract", response_model=MaterialExtractResponse)
async def extract_profile_material(file: UploadFile = File(...)) -> MaterialExtractResponse:
    file_name = Path(file.filename or "upload").name
    data = await file.read(MAX_MATERIAL_BYTES + 1)
    content_type = file.content_type
    await file.close()
    if len(data) > MAX_MATERIAL_BYTES:
        raise HTTPException(status_code=413, detail="材料文件不能超过 20MB。")
    if not data:
        raise HTTPException(status_code=422, detail="材料文件为空。")
    try:
        text, truncated = extract_material_text(file_name, data)
    except MaterialExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    response = MaterialExtractResponse(
        file_name=file_name,
        text=text,
        char_count=len(text),
        truncated=truncated,
    )
    _record_business_event(
        "profile_material",
        "profile.material.extract",
        {
            **_material_metadata(file_name, data, content_type=content_type),
            "char_count": response.char_count,
            "truncated": response.truncated,
            "extractor": "text",
        },
    )
    return response


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
        response = await MultimodalEvidenceExtractor().extract(
            file_name=file_name,
            data=data,
            mime_type=content_type,
        )
    except MultimodalExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LLMGatewayError as exc:
        logger.warning("multimodal evidence extraction failed: %s", exc)
        raise HTTPException(status_code=llm_error_status(exc), detail=str(exc)) from exc
    _record_business_event(
        "profile_material",
        "profile.material.multimodal_extract",
        {
            **_material_metadata(file_name, data, content_type=content_type),
            "model": response.model,
            "page_count": response.page_count,
            "item_count": len(response.items),
            "rejected_count": response.rejected_count,
            "extractor": "qwen-vl",
        },
    )
    return response


@router.get("/cards", response_model=ProfileCardsResponse)
def get_profile_cards() -> ProfileCardsResponse:
    return _profile_store().get_profile()


@router.get("/overview", response_model=ProfileOverviewResponse)
def get_profile_overview() -> ProfileOverviewResponse:
    return _profile_store().get_profile_overview()


@router.get("/conversations")
def get_profile_conversations(
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict[str, object]]:
    """Return the authenticated user's formal exploration turns for audit review."""

    context = get_request_context()
    if context.app_mode != "use" or not context.user_id:
        raise HTTPException(status_code=401, detail="请先登录后再查看对话记录。")
    return ConversationStore(get_settings().profile_db_path).recent(
        user_id=context.user_id,
        limit=limit,
    )


@router.post("/cards/confirm", response_model=ProfileCardsResponse)
def confirm_profile_cards(request: ConfirmProfileCardsRequest) -> ProfileCardsResponse:
    response = _profile_store().confirm_cards(request.cards, request.trace_id)
    _record_business_event(
        "profile_cards",
        "profile.cards.confirm",
        {
            "card_ids": [card.id for card in request.cards],
            "card_count": len(request.cards),
            "trace_id": request.trace_id,
            "profile_version": response.version,
        },
    )
    return response


@router.patch("/cards/{card_id}", response_model=ProfileCardsResponse)
def update_profile_card(
    card_id: str,
    request: ProfileCardPatchRequest,
) -> ProfileCardsResponse:
    try:
        response = _profile_store().update_card(
            card_id,
            request,
            trace_id=get_request_context().request_id,
        )
        _record_business_event(
            "profile_cards",
            "profile.cards.update",
            {
                "card_id": card_id,
                "changed_fields": sorted(request.model_dump(exclude_unset=True)),
                "profile_version": response.version,
            },
        )
        return response
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="这张能力卡不存在。") from exc


@router.delete("/cards/{card_id}", response_model=ProfileCardsResponse)
def delete_profile_card(card_id: str) -> ProfileCardsResponse:
    try:
        response = _profile_store().delete_card(
            card_id,
            trace_id=get_request_context().request_id,
        )
        _record_business_event(
            "profile_cards",
            "profile.cards.delete",
            {"card_id": card_id, "profile_version": response.version},
        )
        return response
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
                response = ProfileProposalResponse.model_validate(cached)
                _record_business_event(
                    "profile_proposal",
                    "profile.proposal.generate",
                    {
                        "trace_id": response.trace_id,
                        "card_count": len(response.card_proposals),
                        "cached": True,
                    },
                )
                return response
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
            _record_business_event(
                "profile_proposal",
                "profile.proposal.generate",
                {
                    "trace_id": response.trace_id,
                    "card_count": len(response.card_proposals),
                    "cached": False,
                },
            )
            return response
        except LLMGatewayError as exc:
            logger.warning("profile proposal failed trace_id=%s reason=%s", trace_id, exc)
            raise HTTPException(status_code=llm_error_status(exc), detail=str(exc)) from exc
        except (ValueError, TypeError) as exc:
            logger.warning("profile proposal invalid trace_id=%s reason=%s", trace_id, exc)
            raise HTTPException(status_code=502, detail="模型输出未通过结构化校验，请稍后重试。") from exc
