import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.agents.profile_agent import ProfileAgent
from app.config import get_settings
from app.schemas.profile import (
    ConfirmProfileCardsRequest,
    ProfileCardPatchRequest,
    ProfileCardsResponse,
    ProfileProposalRequest,
    ProfileProposalResponse,
)
from app.services.llm_gateway import DashScopeQwenGateway, LLMGatewayError
from app.services.profile_store import ProfileStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/profile", tags=["profile"])


def _profile_store() -> ProfileStore:
    return ProfileStore(get_settings().profile_db_path)


@router.get("/cards", response_model=ProfileCardsResponse)
def get_profile_cards() -> ProfileCardsResponse:
    return _profile_store().get_profile()


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
        raise HTTPException(status_code=404, detail="画像卡不存在。") from exc


@router.delete("/cards/{card_id}", response_model=ProfileCardsResponse)
def delete_profile_card(card_id: str) -> ProfileCardsResponse:
    try:
        return _profile_store().delete_card(card_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="画像卡不存在。") from exc


@router.post("/proposals", response_model=ProfileProposalResponse)
async def create_profile_proposal(
    request: ProfileProposalRequest,
) -> ProfileProposalResponse:
    trace_id = uuid4().hex
    agent = ProfileAgent(DashScopeQwenGateway(get_settings()))
    try:
        return await agent.propose(request, trace_id)
    except LLMGatewayError as exc:
        logger.warning("profile proposal failed trace_id=%s reason=%s", trace_id, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        logger.warning("profile proposal invalid trace_id=%s reason=%s", trace_id, exc)
        raise HTTPException(status_code=502, detail="模型输出未通过结构化校验，请稍后重试。") from exc
