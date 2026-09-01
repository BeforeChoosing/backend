from __future__ import annotations

import hashlib
import os
import time

from fastapi import APIRouter, Header, HTTPException

from app.config import get_settings
from app.schemas.auth import (
    AuthLoginRequest,
    AuthLogoutResponse,
    AuthRegisterRequest,
    AuthResponse,
    AuthUser,
    PasswordResetConfirmRequest,
    PasswordResetRequest,
    PasswordResetResponse,
)
from app.services.audit_log import AuditLogStore
from app.services.auth_store import AuthStore
from app.services.request_context import get_request_context
from app.services.password_reset_email import send_password_reset_email


router = APIRouter(prefix="/auth", tags=["auth"])
_reset_cooldowns: dict[str, float] = {}


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


@router.post("/password-reset/request", response_model=PasswordResetResponse, status_code=202)
def request_password_reset(request: PasswordResetRequest) -> PasswordResetResponse:
    # Keep this response identical for known and unknown addresses.
    try:
        email = AuthStore.normalize_email(request.email)
    except ValueError:
        return PasswordResetResponse(detail="如果邮箱已注册，验证码将发送到你的邮箱。")
    now = time.monotonic()
    cooldown = max(30, int(os.getenv("PASSWORD_RESET_COOLDOWN_SECONDS", "60")))
    if now - _reset_cooldowns.get(email, 0) < cooldown:
        return PasswordResetResponse(detail="如果邮箱已注册，验证码将发送到你的邮箱。")
    _reset_cooldowns[email] = now
    try:
        payload = _store().create_password_reset_code(email)
        if payload is not None:
            recipient, code = payload
            if not send_password_reset_email(recipient, code):
                _record_auth_event(action="auth.password_reset.request", result="delivery_failed", email=email)
            else:
                _record_auth_event(action="auth.password_reset.request", result="sent", email=email)
    except ValueError:
        pass
    return PasswordResetResponse(detail="如果邮箱已注册，验证码将发送到你的邮箱。")


@router.post("/password-reset/confirm", response_model=PasswordResetResponse)
def confirm_password_reset(request: PasswordResetConfirmRequest) -> PasswordResetResponse:
    try:
        _store().reset_password(request.email, request.code, request.new_password)
    except ValueError as exc:
        _record_auth_event(action="auth.password_reset.confirm", result="rejected", email=request.email)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _record_auth_event(action="auth.password_reset.confirm", result="success", email=request.email)
    return PasswordResetResponse(detail="密码已重置，请使用新密码登录。")


@router.get("/me", response_model=AuthUser)
def current_user() -> AuthUser:
    context = get_request_context()
    if not context.user_id or not context.user_email:
        raise HTTPException(status_code=401, detail="请先登录后再继续。")
    return AuthUser(id=context.user_id, email=context.user_email, display_name=context.user_name)


@router.post("/logout", response_model=AuthLogoutResponse)
def logout(authorization: str = Header(default="")) -> AuthLogoutResponse:
    _, _, token = authorization.partition(" ")
    token = token.strip()
    _store().revoke_token(token)
    context = get_request_context()
    _record_auth_event(action="auth.logout", result="success", user_id=context.user_id, email=context.user_email)
    return AuthLogoutResponse()
