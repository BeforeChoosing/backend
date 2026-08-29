from __future__ import annotations

import hashlib

from fastapi import APIRouter, Header, HTTPException

from app.config import get_settings
from app.schemas.auth import (
    AuthLoginRequest,
    AuthLogoutResponse,
    AuthRegisterRequest,
    AuthResponse,
    AuthUser,
)
from app.services.audit_log import AuditLogStore
from app.services.auth_store import AuthStore
from app.services.request_context import get_request_context


router = APIRouter(prefix="/auth", tags=["auth"])


def _store() -> AuthStore:
    settings = get_settings()
    return AuthStore(settings.profile_db_path, session_ttl_hours=settings.auth_session_ttl_hours)


def _response(payload: dict[str, object]) -> AuthResponse:
    return AuthResponse.model_validate(payload)


def _account_fingerprint(email: str) -> str:
    return hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()[:16]


def _record_auth_event(
    *,
    action: str,
    result: str,
    user_id: str = "",
    email: str = "",
) -> None:
    settings = get_settings()
    context = get_request_context()
    if context.app_mode != "use":
        return
    try:
        AuditLogStore(settings.profile_db_path).record(
            event_type="auth_event",
            app_mode="use",
            request_id=context.request_id,
            action=action,
            user_id=user_id,
            metadata={
                "result": result,
                "account_fingerprint": _account_fingerprint(email) if email else None,
            },
        )
    except Exception:
        # Authentication must remain available if local audit storage is unavailable.
        pass


@router.post("/register", response_model=AuthResponse, status_code=201)
def register(request: AuthRegisterRequest) -> AuthResponse:
    try:
        payload = _store().register(
            request.email,
            request.password,
            request.display_name,
        )
    except ValueError as exc:
        _record_auth_event(action="auth.register", result="rejected", email=request.email)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    user = AuthUser.model_validate(payload["user"])
    _record_auth_event(action="auth.register", result="success", user_id=user.id, email=user.email)
    return _response({**payload, "user": user})


@router.post("/login", response_model=AuthResponse)
def login(request: AuthLoginRequest) -> AuthResponse:
    try:
        payload = _store().login(request.email, request.password)
    except ValueError as exc:
        _record_auth_event(action="auth.login", result="rejected", email=request.email)
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    user = AuthUser.model_validate(payload["user"])
    _record_auth_event(action="auth.login", result="success", user_id=user.id, email=user.email)
    return _response({**payload, "user": user})


@router.get("/me", response_model=AuthUser)
def current_user() -> AuthUser:
    context = get_request_context()
    if not context.user_id or not context.user_email:
        raise HTTPException(status_code=401, detail="请先登录后再继续。")
    return AuthUser(id=context.user_id, email=context.user_email, display_name=context.user_name)


@router.post("/logout", response_model=AuthLogoutResponse)
def logout(authorization: str = Header(default="")) -> AuthLogoutResponse:
    token = authorization.removeprefix("Bearer ").strip()
    _store().revoke_token(token)
    context = get_request_context()
    _record_auth_event(action="auth.logout", result="success", user_id=context.user_id, email=context.user_email)
    return AuthLogoutResponse()
