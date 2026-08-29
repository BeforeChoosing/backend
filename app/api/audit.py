from __future__ import annotations

from fastapi import APIRouter, Header, Query

from app.config import get_settings
from app.schemas.audit import AuditEventRequest, AuditEventResponse, AuditUsageSummary
from app.services.audit_log import AuditLogStore


router = APIRouter(prefix="/audit", tags=["audit"])


def _store() -> AuditLogStore:
    return AuditLogStore(get_settings().profile_db_path)


@router.post("/events", response_model=AuditEventResponse, status_code=202)
def create_audit_event(
    request: AuditEventRequest,
    x_app_mode: str = Header(default="unknown"),
    x_client_request_id: str = Header(default=""),
) -> AuditEventResponse:
    if x_app_mode.strip().lower() != "use":
        return AuditEventResponse(recorded=False)
    event_id = _store().record(
        event_type="ui_action",
        app_mode="use",
        request_id=x_client_request_id,
        action=request.action,
        metadata={"target": request.target, **request.metadata},
    )
    return AuditEventResponse(recorded=True, event_id=event_id)


@router.get("/usage", response_model=AuditUsageSummary)
def get_audit_usage(x_app_mode: str = Header(default="unknown")) -> AuditUsageSummary:
    mode = "use" if x_app_mode.strip().lower() == "use" else "unknown"
    return AuditUsageSummary.model_validate(_store().usage_summary(app_mode=mode))


@router.get("/events")
def list_audit_events(
    limit: int = Query(default=100, ge=1, le=500),
    x_app_mode: str = Header(default="unknown"),
) -> list[dict]:
    mode = "use" if x_app_mode.strip().lower() == "use" else "unknown"
    return _store().recent(limit=limit, app_mode=mode)
