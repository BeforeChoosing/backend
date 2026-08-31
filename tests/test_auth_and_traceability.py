from __future__ import annotations

from pathlib import Path
from dataclasses import replace

from fastapi.testclient import TestClient

from app import main as main_module
from app.api import audit as audit_api
from app.api import auth as auth_api
from app.api import profile as profile_api
from app.schemas.profile import ProfileExplorationResponse
from app.config import Settings
from app.services.audit_log import AuditLogStore, record_business_event
from app.services.auth_store import AuthStore
from app.services.conversation_store import ConversationStore
from app.services.request_context import RequestContext, reset_request_context, set_request_context


def test_auth_store_register_login_and_revoke(tmp_path: Path) -> None:
    store = AuthStore(tmp_path / "profile.db")

    registered = store.register("Person@Example.com", "password-123")
    assert registered["user"]["email"] == "person@example.com"
    assert store.resolve_token(str(registered["access_token"]))["id"] == registered["user"]["id"]

    logged_in = store.login("person@example.com", "password-123")
    assert logged_in["user"]["id"] == registered["user"]["id"]
    assert store.resolve_token(str(logged_in["access_token"])) is not None

    assert store.revoke_token(str(logged_in["access_token"])) is True
    assert store.resolve_token(str(logged_in["access_token"])) is None


def test_auth_store_rejects_duplicate_and_bad_password(tmp_path: Path) -> None:
    store = AuthStore(tmp_path / "profile.db")
    store.register("person@example.com", "password-123")

    try:
        store.register("PERSON@example.com", "password-123")
    except ValueError as exc:
        assert "已经注册" in str(exc)
    else:
        raise AssertionError("duplicate account should be rejected")

    try:
        store.login("person@example.com", "wrong-password")
    except ValueError as exc:
        assert "不正确" in str(exc)
    else:
        raise AssertionError("bad password should be rejected")


def test_register_api_rejects_existing_email_case_insensitively(tmp_path: Path, monkeypatch) -> None:
    settings: Settings = replace(main_module.settings, profile_db_path=str(tmp_path / "profile.db"))
    monkeypatch.setattr(main_module, "settings", settings)
    monkeypatch.setattr(auth_api, "get_settings", lambda: settings)
    client = TestClient(main_module.app)
    first = client.post(
        "/api/v1/auth/register",
        json={"email": "Duplicate@Example.com", "password": "password-123"},
    )
    assert first.status_code == 201

    duplicate = client.post(
        "/api/v1/auth/register",
        json={"email": "duplicate@example.com", "password": "password-456"},
    )
    assert duplicate.status_code == 409
    assert "已经注册" in duplicate.json()["detail"]


def test_formal_business_events_and_conversations_are_user_bound(tmp_path: Path) -> None:
    db_path = tmp_path / "profile.db"
    token = set_request_context(
        RequestContext(
            app_mode="use",
            request_id="request-1",
            user_id="user-1",
            user_email="person@example.com",
            user_name="Person",
        )
    )
    try:
        event_id = record_business_event(
            db_path,
            event_type="profile_cards",
            action="profile.cards.delete",
            metadata={"card_id": "card-1"},
        )
        assert event_id
        ConversationStore(db_path).record_turn(
            user_id="user-1",
            request_id="request-1",
            trace_id="trace-1",
            experience_text="我负责过一次用户访谈并调整了方案。",
            messages=[{"role": "user", "content": "我负责访谈。"}],
            response={"reply": "请补充你的判断依据。"},
        )
    finally:
        reset_request_context(token)

    events = AuditLogStore(db_path).recent(user_id="user-1")
    assert events[0]["user_id"] == "user-1"
    assert events[0]["action"] == "profile.cards.delete"
    conversations = ConversationStore(db_path).recent(user_id="user-1")
    assert conversations[0]["trace_id"] == "trace-1"
    assert conversations[0]["messages"][0]["content"] == "我负责访谈。"


def test_demo_context_does_not_write_traceability_records(tmp_path: Path) -> None:
    db_path = tmp_path / "profile.db"
    token = set_request_context(RequestContext(app_mode="demo", request_id="demo"))
    try:
        assert record_business_event(
            db_path,
            event_type="profile_cards",
            action="profile.cards.delete",
        ) is None
    finally:
        reset_request_context(token)
    assert AuditLogStore(db_path).usage_summary(app_mode="use")["event_count"] == 0


def test_formal_api_requires_auth_and_persists_chat_trace(tmp_path: Path, monkeypatch) -> None:
    settings: Settings = replace(main_module.settings, profile_db_path=str(tmp_path / "profile.db"))
    monkeypatch.setattr(main_module, "settings", settings)
    monkeypatch.setattr(audit_api, "get_settings", lambda: settings)
    monkeypatch.setattr(auth_api, "get_settings", lambda: settings)
    monkeypatch.setattr(profile_api, "get_settings", lambda: settings)

    class FakeProfileAgent:
        async def explore(self, request, trace_id):
            return ProfileExplorationResponse(
                trace_id=trace_id,
                reply="请补充这次取舍依据。",
                focus_dimension="decision",
                evidence_found=["负责用户访谈"],
                evidence_gap="仍缺少取舍依据。",
                potential_hypotheses=["具备问题界定潜能。"],
            )

    monkeypatch.setattr(profile_api, "_profile_agent", lambda: FakeProfileAgent())
    client = TestClient(main_module.app)
    formal_headers = {"X-App-Mode": "use"}

    assert client.get("/api/v1/profile/cards", headers=formal_headers).status_code == 401
    assert client.get("/api/v1/profile/cards", headers={"X-App-Mode": "demo"}).status_code == 401
    assert client.get("/api/v1/trial/catalog", headers={"X-App-Mode": "demo"}).status_code == 200

    registered = client.post(
        "/api/v1/auth/register",
        headers=formal_headers,
        json={"email": "trace@example.com", "password": "password-123"},
    )
    assert registered.status_code == 201
    token = registered.json()["access_token"]
    headers = {**formal_headers, "Authorization": f"Bearer {token}"}
    response = client.post(
        "/api/v1/profile/exploration/messages",
        headers=headers,
        json={
            "experience_text": "我在校园项目中负责访谈用户并整理反馈，随后调整了方案。",
            "messages": [{"role": "user", "content": "我负责访谈并整理了反馈。"}],
        },
    )
    assert response.status_code == 200

    conversations = client.get("/api/v1/profile/conversations", headers=headers)
    assert conversations.status_code == 200
    assert conversations.json()[0]["messages"][0]["content"] == "我负责访谈并整理了反馈。"
    events = client.get("/api/v1/audit/events", headers=headers)
    assert events.status_code == 200
    assert any(item["event_type"] == "profile_chat" for item in events.json())
