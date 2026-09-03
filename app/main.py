import re
import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException

from app.api.errors import error_response, http_error_handler, validation_error_handler
from app.services.runtime_log import log_event

from app.api.audit import router as audit_router
from app.api.auth import router as auth_router
from app.api.health import router as health_router
from app.api.llm_queue import router as llm_queue_router
from app.api.career import router as career_router
from app.api.profile import router as profile_router
from app.api.trial import router as trial_router
from app.config import get_settings
from app.services.audit_log import AuditLogStore
from app.services.auth_store import AuthStore
from app.version import APP_VERSION
from app.services.request_context import (
    RequestContext,
    reset_request_context,
    set_request_context,
)

settings = get_settings()
app = FastAPI(title=settings.app_name, version=APP_VERSION)
app.add_exception_handler(HTTPException, http_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)


@app.middleware("http")
async def audit_formal_mode_requests(request: Request, call_next):
    """Authenticate private APIs independently of client-selected UI mode."""

    app_mode = request.headers.get("X-App-Mode", "unknown").strip().lower()
    path = request.url.path.rstrip("/") or "/"
    is_api = path == settings.api_prefix or path.startswith(settings.api_prefix + "/")
    public_auth = request.method == "POST" and path in {
        f"{settings.api_prefix}/auth/login", f"{settings.api_prefix}/auth/register",
        f"{settings.api_prefix}/auth/password-reset/request",
        f"{settings.api_prefix}/auth/password-reset/confirm",
    }
    public_read = request.method in {"GET", "HEAD"} and (
        path == f"{settings.api_prefix}/health"
        or re.fullmatch(
            re.escape(settings.api_prefix) + r"/trial/(?:catalog(?:/[^/]+)?|tasks/[^/]+)",
            path,
        ) is not None
    )
    requires_auth = is_api and not public_auth and not public_read
    # Any private API access is a real-mode operation. A forged/missing mode
    # cannot disable either authentication or audit logging.
    if requires_auth or public_auth:
        app_mode = "use"
    # The server owns the trace ID; client-supplied IDs are only correlation hints.
    request_id = uuid4().hex
    client_request_id = request.headers.get("X-Client-Request-Id", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", client_request_id):
        client_request_id = ""
    response = None
    status_code = 500
    authenticated_user = None
    context = RequestContext(app_mode=app_mode, request_id=request_id)
    token = set_request_context(context)
    started = time.perf_counter()
    try:
        if request.method != "OPTIONS" and requires_auth:
            authorization = request.headers.get("Authorization", "")
            scheme, _, raw_token = authorization.partition(" ")
            if scheme.lower() != "bearer" or not raw_token.strip():
                status_code = 401
                response = error_response(401, "正式模式需要先登录。")
            else:
                authenticated_user = AuthStore(
                    settings.profile_db_path,
                    session_ttl_hours=settings.auth_session_ttl_hours,
                ).resolve_token(raw_token)
                if authenticated_user is None:
                    status_code = 401
                    response = error_response(401, "登录已失效，请重新登录。")
        context = RequestContext(
            app_mode=app_mode, request_id=request_id,
            user_id=str(authenticated_user["id"]) if authenticated_user else "",
            user_email=str(authenticated_user["email"]) if authenticated_user else "",
            user_name=str(authenticated_user["display_name"]) if authenticated_user else "",
        )
        set_request_context(context)
        if response is None:
            response = await call_next(request)
            status_code = response.status_code
        return response
    except Exception as exc:
        log_event('request_failed', level='error', error=exc, alert=True, status_code=500,
                  error_code='INTERNAL_ERROR')
        response = error_response(500)
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
                    user_id=context.user_id,
                    request_id=request_id,
                    action=f"{request.method} {request.url.path}",
                    method=request.method,
                    path=request.url.path,
                    status_code=status_code,
                    duration_ms=duration_ms,
                    metadata={"client_request_id": client_request_id},
                )
            except Exception as exc:  # Audit failure must be visible to operators.
                log_event('audit_write_failed', level='error', error=exc, alert=True)
        if is_api and request.method != 'OPTIONS':
            route = request.scope.get('route')
            log_event('http_request', level='error' if status_code >= 500 else ('warn' if status_code >= 400 else 'info'),
                      method=request.method, route=getattr(route, 'path', 'unmatched'),
                      status_code=status_code, duration_ms=round(duration_ms, 2),
                      client_request_id=client_request_id)
        reset_request_context(token)
        if response is not None:
            response.headers["X-Request-Id"] = request_id
            if requires_auth or public_auth:
                response.headers["Cache-Control"] = "no-store"


# CORS wraps the auth middleware so browsers can receive its 401 responses.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-Id", "Retry-After"],
)


app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(auth_router, prefix=settings.api_prefix)
app.include_router(profile_router, prefix=settings.api_prefix)
app.include_router(career_router, prefix=settings.api_prefix)
app.include_router(trial_router, prefix=settings.api_prefix)
app.include_router(audit_router, prefix=settings.api_prefix)
app.include_router(llm_queue_router, prefix=settings.api_prefix)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": settings.app_name, "docs": "/docs"}
