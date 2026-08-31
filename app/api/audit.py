from __future__ import annotations

from fastapi import APIRouter, Query

from app.config import get_settings
from app.schemas.audit import AuditEventRequest, AuditEventResponse, AuditUsageSummary
from app.services.audit_log import AuditLogStore
from app.services.user_data import require_user_context


router = APIRouter(prefix="/audit", tags=["audit"])


def _store() -> AuditLogStore:
    return AuditLogStore(get_settings().profile_db_path)


@router.post("/events", response_model=AuditEventResponse, status_code=202)
def create_audit_event(
    request: AuditEventRequest,
) -> AuditEventResponse:
    # Middleware derives this from the authenticated request, not the UI header.
    context = require_user_context()
    event_id = _store().record(
        event_type="ui_action",
        app_mode="use",
        user_id=context.user_id,
        request_id=context.request_id,
        action=request.action,
        metadata={"target": request.target, **request.metadata},
    )
    return AuditEventResponse(recorded=True, event_id=event_id)


@router.get("/usage", response_model=AuditUsageSummary)
def get_audit_usage() -> AuditUsageSummary:
    context = require_user_context()
    return AuditUsageSummary.model_validate(
        _store().usage_summary(app_mode="use", user_id=context.user_id)
    )


@router.get("/events")
def list_audit_events(
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict]:
    context = require_user_context()
    return _store().recent(
        limit=limit,
        app_mode="use",
        user_id=context.user_id,
    )
