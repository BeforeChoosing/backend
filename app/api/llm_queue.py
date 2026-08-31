from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.services.llm_request_queue import get_llm_request_queue
from app.services.request_context import get_request_context


router = APIRouter(prefix="/llm-queue", tags=["llm-queue"])


def _queue():
    settings = get_settings()
    return get_llm_request_queue(
        max_concurrency=settings.llm_max_concurrency,
        max_requests_per_minute=settings.llm_max_requests_per_minute,
    )


@router.get("/me")
def current_queue_status() -> dict[str, object]:
    user_id = get_request_context().user_id
    if not user_id:
        raise HTTPException(status_code=401, detail="请先登录后再继续。")
    return _queue().status_for_user(user_id)


@router.delete("/me")
def cancel_current_request() -> dict[str, object]:
    user_id = get_request_context().user_id
    if not user_id:
        raise HTTPException(status_code=401, detail="请先登录后再继续。")
    return {"cancelled": _queue().cancel_for_user(user_id)}
