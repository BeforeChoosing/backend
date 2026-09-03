import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from app.agents.profile_agent import ProfileAgent
from app.config import get_settings
from app.schemas.profile import (
    MaterialUnderstandingRequest,
    MaterialUnderstandingResponse,
    ConfirmProfileCardsRequest,
    MaterialExtractResponse,
    ProfileExplorationRequest,
    ProfileExplorationResponse,
    ProfileCardPatchRequest,
    ProfileCardsResponse,
    ProfileConversationSnapshot,
    ProfileConversationSnapshotUpsert,
    ProfileOverviewResponse,
    ProfileMemoryResetRequest,
    ProfileMemoryResetResponse,
    ProfileProposalRequest,
    ProfileProposalResponse,
)
from app.schemas.multimodal import MultimodalEvidenceResponse
from app.services.llm_gateway import DashScopeQwenGateway, LLMGatewayError, llm_error_status
from app.services.json_stream import JsonStringFieldAccumulator
from app.services.material_extractor import MaterialExtractionError, extract_material_text
from app.services.material_store import MaterialStorageError, MaterialStore
from app.services.model_response_cache import ModelResponseCache
from app.services.multimodal_evidence import MultimodalEvidenceExtractor, MultimodalExtractionError
from app.services.audit_log import record_business_event
from app.services.conversation_store import ConversationStore
from app.services.request_context import (
    get_request_context,
    reset_request_context,
    set_request_context,
)
from app.services.profile_exploration_controller import (
    CONTROLLER_VERSION,
    apply_exploration_controller,
)
from app.services.profile_store import ProfileStore
from app.services.user_data import user_data_path
from app.services.user_memory import reset_user_memory
from app.services.llm_request_queue import get_llm_request_queue
from app.version import APP_VERSION

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/profile", tags=["profile"])
MAX_MATERIAL_BYTES = 20 * 1024 * 1024
_exploration_locks: dict[str, asyncio.Lock] = {}
_proposal_locks: dict[str, asyncio.Lock] = {}
_material_understanding_locks: dict[str, asyncio.Lock] = {}
_CONVERSATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,120}$")


def _profile_store() -> ProfileStore:
    return ProfileStore(user_data_path(get_settings().profile_db_path))


def _profile_agent() -> ProfileAgent:
    return ProfileAgent(DashScopeQwenGateway(get_settings()))


def _material_store() -> MaterialStore:
    return MaterialStore(user_data_path(get_settings().profile_db_path))


def _formal_user_id() -> str:
    context = get_request_context()
    if context.app_mode != "use" or not context.user_id:
        raise HTTPException(status_code=401, detail="请先登录后再同步对话记录。")
    return context.user_id


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


@router.post("/materials/understand", response_model=MaterialUnderstandingResponse)
async def understand_profile_material(
    request: MaterialUnderstandingRequest,
) -> MaterialUnderstandingResponse:
    """Turn extracted/OCR text into user-selectable experience candidates.

    Text extraction and visual OCR deliberately remain separate from this
    step.  The second, small model call gives the user a concrete choice of
    which experience to discuss instead of ending the interaction after OCR.
    """

    cache_key = ModelResponseCache.fingerprint(
        {
            "prompt_version": ProfileAgent.MATERIAL_PROMPT_VERSION,
            "file_name": request.file_name,
            "text": request.text,
            "stored_material_id": request.stored_material_id,
        }
    )
    lock = _material_understanding_locks.setdefault(cache_key, asyncio.Lock())
    async with lock:
        cached = _model_cache().get("profile-material-understanding", cache_key)
        if cached is not None:
            try:
                response = MaterialUnderstandingResponse.model_validate(cached)
                return response.model_copy(update={"cache_hit": True})
            except ValueError:
                logger.warning("discarding invalid cached material understanding")
        trace_id = uuid4().hex
        try:
            response = await _profile_agent().understand_material(request, trace_id)
            response = response.model_copy(update={"cache_hit": False})
            _model_cache().set(
                "profile-material-understanding",
                cache_key,
                response.model_dump(mode="json"),
            )
            _record_business_event(
                "profile_material",
                "profile.material.understand",
                {
                    "trace_id": response.trace_id,
                    "file_name": request.file_name,
                    "candidate_count": len(response.experience_candidates),
                    "model": response.model,
                    "cached": False,
                },
            )
            return response
        except LLMGatewayError as exc:
            logger.warning("material understanding failed trace_id=%s reason=%s", trace_id, exc)
            raise HTTPException(status_code=llm_error_status(exc), detail=str(exc)) from exc
        except (ValueError, TypeError) as exc:
            logger.warning("material understanding invalid trace_id=%s reason=%s", trace_id, exc)
            raise HTTPException(status_code=502, detail="材料经历整理没有完成，请稍后重试。") from exc


@router.post("/exploration/messages", response_model=ProfileExplorationResponse)
async def create_profile_exploration_message(
    request: ProfileExplorationRequest,
) -> ProfileExplorationResponse:
    cache_key = ModelResponseCache.fingerprint(
        {
            "prompt_version": ProfileAgent.EXPLORATION_PROMPT_VERSION,
            "controller_version": CONTROLLER_VERSION,
            "model_pool": request.model_tier,
            "experience_text": request.experience_text,
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "focus_history": request.focus_history,
            "target_role": request.target_role,
            "existing_card_titles": request.existing_card_titles,
            "model_tier": request.model_tier,
            "round_number": request.round_number,
            "star_history": request.star_history,
            "stop_requested": request.stop_requested,
        }
    )
    lock = _exploration_locks.setdefault(cache_key, asyncio.Lock())
    async with lock:
        cached = _model_cache().get("profile-exploration", cache_key)
        if cached is not None:
            try:
                response = ProfileExplorationResponse.model_validate(cached)
                response = response.model_copy(update={"cache_hit": True})
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
            response = response.model_copy(update={"cache_hit": False})
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


def _stream_event(event: str, payload: object) -> str:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
    )


@router.post("/exploration/messages/stream")
async def stream_profile_exploration_message(
    request: ProfileExplorationRequest,
) -> StreamingResponse:
    """Stream the reply field while retaining the validated final response."""

    settings = get_settings()
    context = get_request_context()
    cache_key = ModelResponseCache.fingerprint(
        {
            "prompt_version": ProfileAgent.EXPLORATION_PROMPT_VERSION,
            "controller_version": CONTROLLER_VERSION,
            "model_pool": request.model_tier,
            "experience_text": request.experience_text,
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "focus_history": request.focus_history,
            "target_role": request.target_role,
            "existing_card_titles": request.existing_card_titles,
            "model_tier": request.model_tier,
            "round_number": request.round_number,
            "star_history": request.star_history,
            "stop_requested": request.stop_requested,
        }
    )
    lock = _exploration_locks.setdefault(cache_key, asyncio.Lock())

    async def events():
        context_token = set_request_context(context)
        try:
            async with lock:
                cached = _model_cache().get("profile-exploration", cache_key)
                if cached is not None:
                    try:
                        response = ProfileExplorationResponse.model_validate(cached)
                        response = response.model_copy(update={"cache_hit": True})
                        _record_exploration_turn(request, response)
                        yield _stream_event("delta", {"text": response.reply})
                        yield _stream_event("done", response.model_dump(mode="json"))
                        return
                    except ValueError:
                        logger.warning("discarding invalid cached profile exploration stream")

                trace_id = uuid4().hex
                event_queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
                loop = asyncio.get_running_loop()
                reply_accumulator = JsonStringFieldAccumulator("reply")

                def emit_model_delta(raw_delta: str) -> None:
                    reply_delta = reply_accumulator.feed(raw_delta)
                    if reply_delta:
                        loop.call_soon_threadsafe(
                            event_queue.put_nowait, ("delta", {"text": reply_delta})
                        )

                def reset_model_delta() -> None:
                    nonlocal reply_accumulator
                    reply_accumulator = JsonStringFieldAccumulator("reply")
                    loop.call_soon_threadsafe(
                        event_queue.put_nowait, ("reset", {})
                    )

                def produce() -> None:
                    worker_token = set_request_context(context)
                    try:
                        response = apply_exploration_controller(
                            _profile_agent().explore_stream(
                                request,
                                trace_id,
                                on_delta=emit_model_delta,
                                on_reset=reset_model_delta,
                            ),
                            request,
                        )
                        response = response.model_copy(update={"cache_hit": False})
                        _model_cache().set(
                            "profile-exploration",
                            cache_key,
                            response.model_dump(mode="json"),
                        )
                        _record_exploration_turn(request, response)
                        loop.call_soon_threadsafe(
                            event_queue.put_nowait,
                            ("done", response.model_dump(mode="json")),
                        )
                    except LLMGatewayError as exc:
                        logger.warning(
                            "profile exploration stream failed trace_id=%s reason=%s",
                            trace_id,
                            exc,
                        )
                        loop.call_soon_threadsafe(
                            event_queue.put_nowait,
                            (
                                "error",
                                {
                                    "message": str(exc),
                                    "status": llm_error_status(exc),
                                    "request_id": context.request_id,
                                },
                            ),
                        )
                    except (ValueError, TypeError) as exc:
                        logger.warning(
                            "profile exploration stream invalid trace_id=%s reason=%s",
                            trace_id,
                            exc,
                        )
                        loop.call_soon_threadsafe(
                            event_queue.put_nowait,
                            (
                                "error",
                                {
                                    "message": "本轮补充引导没有整理成功，请稍后重试。",
                                    "status": 502,
                                    "request_id": context.request_id,
                                },
                            ),
                        )
                    except Exception as exc:  # noqa: BLE001 - stream already started
                        logger.exception(
                            "unexpected profile exploration stream failure trace_id=%s",
                            trace_id,
                        )
                        loop.call_soon_threadsafe(
                            event_queue.put_nowait,
                            (
                                "error",
                                {
                                    "message": "服务暂时无法完成这次回复。",
                                    "status": 500,
                                    "request_id": context.request_id,
                                },
                            ),
                        )
                    finally:
                        reset_request_context(worker_token)

                producer = asyncio.create_task(asyncio.to_thread(produce))
                try:
                    while True:
                        event, payload = await event_queue.get()
                        yield _stream_event(event, payload)
                        if event in {"done", "error"}:
                            break
                    await producer
                finally:
                    if not producer.done():
                        producer.cancel()
        finally:
            reset_request_context(context_token)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


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
    try:
        stored = _material_store().save(
            original_name=file_name,
            data=data,
            mime_type=content_type or "application/octet-stream",
        )
    except MaterialStorageError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    response = MaterialExtractResponse(
        file_name=file_name,
        text=text,
        char_count=len(text),
        truncated=truncated,
        stored_material_id=str(stored["id"]),
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
    try:
        stored = _material_store().save(
            original_name=file_name,
            data=data,
            mime_type=content_type or "application/octet-stream",
        )
    except MaterialStorageError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    response = response.model_copy(update={"stored_material_id": str(stored["id"])})
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


@router.get("/materials/{material_id}")
def download_profile_material(material_id: str) -> FileResponse:
    if not re.fullmatch(r"material-[a-f0-9]{32}", material_id):
        raise HTTPException(status_code=404, detail="该材料不存在。")
    stored = _material_store().get(material_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="该材料不存在。")
    metadata, path = stored
    return FileResponse(
        path,
        media_type=str(metadata["mime_type"]),
        filename=str(metadata["original_name"]),
        headers={"Cache-Control": "private, no-store"},
    )


@router.get("/cards", response_model=ProfileCardsResponse)
def get_profile_cards() -> ProfileCardsResponse:
    return _profile_store().get_profile()


@router.get("/overview", response_model=ProfileOverviewResponse)
def get_profile_overview() -> ProfileOverviewResponse:
    return _profile_store().get_profile_overview()


@router.delete("/memory", response_model=ProfileMemoryResetResponse)
def delete_profile_memory(
    request: ProfileMemoryResetRequest,
) -> ProfileMemoryResetResponse:
    """Clear product memory while preserving the account and audit trail."""

    del request  # Literal validation above is the destructive-action confirmation.
    context = get_request_context()
    user_id = _formal_user_id()
    settings = get_settings()
    queue = get_llm_request_queue(
        max_concurrency=settings.llm_max_concurrency,
        max_requests_per_minute=settings.llm_max_requests_per_minute,
        model_max_concurrency=settings.llm_model_max_concurrency,
    )
    cancelled_requests = queue.cancel_all_for_user(user_id)
    removed_records, removed_files = reset_user_memory(
        user_data_path(settings.profile_db_path),
        account_db_path=settings.profile_db_path,
        user_id=user_id,
    )
    _record_business_event(
        "profile_memory",
        "profile.memory.reset",
        {
            "removed_records": removed_records,
            "removed_files": removed_files,
            "cancelled_requests": cancelled_requests,
            "request_id": context.request_id,
        },
    )
    return ProfileMemoryResetResponse(
        removed_records=removed_records,
        removed_files=removed_files,
        cancelled_requests=cancelled_requests,
        version=APP_VERSION,
    )


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


@router.get(
    "/conversation-snapshots",
    response_model=list[ProfileConversationSnapshot],
)
def list_profile_conversation_snapshots(
    limit: int = Query(default=50, ge=1, le=50),
) -> list[ProfileConversationSnapshot]:
    """Return formal-mode conversations for the authenticated account."""

    user_id = _formal_user_id()
    snapshots = ConversationStore(get_settings().profile_db_path).list_snapshots(
        user_id=user_id,
        limit=limit,
    )
    return [ProfileConversationSnapshot.model_validate(item) for item in snapshots]


@router.put(
    "/conversation-snapshots/{conversation_id}",
    response_model=ProfileConversationSnapshot,
)
def upsert_profile_conversation_snapshot(
    conversation_id: str,
    request: ProfileConversationSnapshotUpsert,
) -> ProfileConversationSnapshot:
    """Create or replace one conversation snapshot for the current account."""

    if not _CONVERSATION_ID_PATTERN.fullmatch(conversation_id):
        raise HTTPException(status_code=422, detail="对话编号格式无效。")
    user_id = _formal_user_id()
    snapshot = ConversationStore(get_settings().profile_db_path).upsert_snapshot(
        user_id=user_id,
        conversation_id=conversation_id,
        snapshot=request.model_dump(mode="json"),
        max_per_user=50,
    )
    return ProfileConversationSnapshot.model_validate(snapshot)


@router.delete("/conversation-snapshots/{conversation_id}")
def delete_profile_conversation_snapshot(conversation_id: str) -> dict[str, bool]:
    if not _CONVERSATION_ID_PATTERN.fullmatch(conversation_id):
        raise HTTPException(status_code=422, detail="对话编号格式无效。")
    user_id = _formal_user_id()
    deleted = ConversationStore(get_settings().profile_db_path).delete_snapshot(
        user_id=user_id,
        conversation_id=conversation_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="该历史对话不存在。")
    return {"deleted": True}


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
    existing_cards = _profile_store().get_profile().cards
    request = ProfileProposalRequest.model_validate({
        **request.model_dump(mode="json"),
        "existing_card_titles": [card.title for card in existing_cards],
        "existing_cards": [
            {
                "id": card.id,
                "title": card.title,
                "category": card.category,
                "description": card.description,
            }
            for card in existing_cards
        ],
    })
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
