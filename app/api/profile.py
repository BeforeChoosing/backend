import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.agents.profile_agent import ProfileAgent
from app.config import get_settings
from app.schemas.profile import ProfileProposalRequest, ProfileProposalResponse
from app.services.llm_gateway import DashScopeQwenGateway, LLMGatewayError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/profile", tags=["profile"])


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
