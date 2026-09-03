from fastapi import APIRouter

from app.config import get_settings
from app.version import APP_VERSION

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, object]:
    settings = get_settings()
    return {
        "status": "ok",
        "version": APP_VERSION,
        "service": settings.app_name,
        "qwen_configured": settings.qwen_configured,
        "model": settings.qwen_model,
        "fast_model": settings.qwen_fast_model,
    }
