from pathlib import Path

from app.services.audit_log import AuditLogStore, record_model_call
from app.services.request_context import RequestContext, reset_request_context, set_request_context


def test_audit_store_records_usage_and_model_calls(tmp_path: Path):
    db_path = tmp_path / "profile.db"
    token = set_request_context(RequestContext(app_mode="use", request_id="req-1"))
    try:
        record_model_call(
            db_path,
            service="qwen",
            model="qwen-plus",
            duration_ms=12.5,
            input_tokens=10,
            output_tokens=4,
        )
    finally:
        reset_request_context(token)
    store = AuditLogStore(db_path)
    store.record(event_type="http_request", app_mode="use", action="POST /trial", duration_ms=3)
    summary = store.usage_summary(app_mode="use")
    assert summary["event_count"] == 2
    assert summary["model_call_count"] == 1
    assert summary["input_tokens"] == 10
    assert summary["output_tokens"] == 4
    assert summary["model_mean_duration_ms"] == 12.5


def test_demo_context_does_not_record_model_call(tmp_path: Path):
    db_path = tmp_path / "profile.db"
    token = set_request_context(RequestContext(app_mode="demo", request_id="demo"))
    try:
        record_model_call(db_path, service="qwen", model="qwen-plus", duration_ms=1)
    finally:
        reset_request_context(token)
    assert AuditLogStore(db_path).usage_summary(app_mode="use")["event_count"] == 0
