import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.audit import router as audit_router
from app.api.health import router as health_router
from app.api.career import router as career_router
from app.api.profile import router as profile_router
from app.api.trial import router as trial_router
from app.config import get_settings
from app.services.audit_log import AuditLogStore
from app.services.request_context import (
    RequestContext,
    reset_request_context,
    set_request_context,
)

settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def audit_formal_mode_requests(request: Request, call_next):
    """Persist request metadata for every formal-mode web operation."""

    app_mode = request.headers.get("X-App-Mode", "unknown").strip().lower()
    request_id = request.headers.get("X-Client-Request-Id", "").strip() or uuid4().hex
    token = set_request_context(RequestContext(app_mode=app_mode, request_id=request_id))
    started = time.perf_counter()
    response = None
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration_ms = (time.perf_counter() - started) * 1000
        if (
            app_mode == "use"
            and request.method != "OPTIONS"
            and not request.url.path.endswith("/health")
        ):
            try:
                AuditLogStore(settings.profile_db_path).record(
                    event_type="http_request",
                    app_mode="use",
                    request_id=request_id,
                    action=f"{request.method} {request.url.path}",
                    method=request.method,
                    path=request.url.path,
                    status_code=status_code,
                    duration_ms=duration_ms,
                    metadata={"query": request.url.query[:400]},
                )
            except Exception:  # noqa: BLE001 - audit must never break the product
                pass
        reset_request_context(token)
        if response is not None:
            response.headers["X-Request-Id"] = request_id


app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(profile_router, prefix=settings.api_prefix)
app.include_router(career_router, prefix=settings.api_prefix)
app.include_router(trial_router, prefix=settings.api_prefix)
app.include_router(audit_router, prefix=settings.api_prefix)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": settings.app_name, "docs": "/docs"}
