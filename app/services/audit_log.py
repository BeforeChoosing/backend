"""Local audit and usage log for formal-mode operations.

The audit table stores who performed an operation, when it happened, the
request trace and bounded metadata. Large or sensitive payloads are kept in
their dedicated local business tables rather than being copied into metadata.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


class AuditLogStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    app_mode TEXT NOT NULL,
                    user_id TEXT,
                    request_id TEXT,
                    action TEXT NOT NULL,
                    method TEXT,
                    path TEXT,
                    status_code INTEGER,
                    duration_ms REAL,
                    model TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(audit_events)").fetchall()
            }
            if "user_id" not in columns:
                connection.execute("ALTER TABLE audit_events ADD COLUMN user_id TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_events_created_at ON audit_events(created_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_events_user ON audit_events(user_id)"
            )

    def record(
        self,
        *,
        event_type: str,
        app_mode: str,
        action: str,
        user_id: str = "",
        request_id: str = "",
        method: str | None = None,
        path: str | None = None,
        status_code: int | None = None,
        duration_ms: float | None = None,
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        event_id = uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        safe_metadata = dict(metadata or {})
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO audit_events (
                    id, created_at, event_type, app_mode, user_id, request_id, action,
                    method, path, status_code, duration_ms, model,
                    input_tokens, output_tokens, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    created_at,
                    event_type[:80],
                    app_mode[:20],
                    user_id[:120] if user_id else None,
                    request_id[:120],
                    action[:200],
                    method[:12] if method else None,
                    path[:240] if path else None,
                    status_code,
                    round(float(duration_ms), 3) if duration_ms is not None else None,
                    model[:120] if model else None,
                    input_tokens,
                    output_tokens,
                    json.dumps(safe_metadata, ensure_ascii=False, default=str)[:4000],
                ),
            )
        return event_id

    def recent(
        self,
        *,
        limit: int = 100,
        app_mode: str = "use",
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 500))
        with self._connect() as connection:
            if user_id:
                rows = connection.execute(
                    """
                    SELECT id, created_at, event_type, app_mode, user_id, request_id, action,
                           method, path, status_code, duration_ms, model,
                           input_tokens, output_tokens, metadata_json
                    FROM audit_events
                    WHERE app_mode = ? AND user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (app_mode, user_id, bounded_limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT id, created_at, event_type, app_mode, user_id, request_id, action,
                           method, path, status_code, duration_ms, model,
                           input_tokens, output_tokens, metadata_json
                    FROM audit_events
                    WHERE app_mode = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (app_mode, bounded_limit),
                ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["metadata"] = json.loads(item.pop("metadata_json"))
            except (TypeError, json.JSONDecodeError):
                item["metadata"] = {}
            result.append(item)
        return result

    def usage_summary(
        self,
        *,
        app_mode: str = "use",
        user_id: str | None = None,
    ) -> dict[str, Any]:
        where = "app_mode = ?"
        params: list[Any] = [app_mode]
        if user_id:
            where += " AND user_id = ?"
            params.append(user_id)
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT COUNT(*) AS event_count,
                       AVG(duration_ms) AS mean_duration_ms,
                       SUM(CASE WHEN event_type = 'model_call' THEN 1 ELSE 0 END) AS model_call_count,
                       AVG(CASE WHEN event_type = 'model_call' THEN duration_ms END) AS model_mean_duration_ms,
                       SUM(COALESCE(input_tokens, 0)) AS input_tokens,
                       SUM(COALESCE(output_tokens, 0)) AS output_tokens
                FROM audit_events WHERE {where}
                """,
                params,
            ).fetchone()
            path_params = list(params)
            paths = connection.execute(
                f"""
                SELECT path, COUNT(*) AS count, AVG(duration_ms) AS mean_duration_ms
                FROM audit_events
                WHERE {where} AND event_type = 'http_request'
                GROUP BY path ORDER BY count DESC
                """,
                path_params,
            ).fetchall()
        return {
            "app_mode": app_mode,
            "event_count": int(row["event_count"] or 0),
            "mean_duration_ms": round(float(row["mean_duration_ms"] or 0), 3),
            "model_mean_duration_ms": round(float(row["model_mean_duration_ms"] or 0), 3),
            "model_call_count": int(row["model_call_count"] or 0),
            "input_tokens": int(row["input_tokens"] or 0),
            "output_tokens": int(row["output_tokens"] or 0),
            "paths": [
                {
                    "path": item["path"],
                    "count": int(item["count"]),
                    "mean_duration_ms": round(float(item["mean_duration_ms"] or 0), 3),
                }
                for item in paths
            ],
        }


def record_model_call(
    db_path: str | Path,
    *,
    service: str,
    model: str,
    duration_ms: float,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Record a model call only for a formal-mode request context."""

    from app.services.request_context import get_request_context

    context = get_request_context()
    if context.app_mode != "use":
        return
    AuditLogStore(db_path).record(
        event_type="model_call",
        app_mode=context.app_mode,
        user_id=context.user_id,
        request_id=context.request_id,
        action=f"{service}:{model}",
        duration_ms=duration_ms,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        metadata=metadata,
    )


def record_business_event(
    db_path: str | Path,
    *,
    event_type: str,
    action: str,
    metadata: Mapping[str, Any] | None = None,
    status_code: int = 200,
) -> str | None:
    """Record a formal-mode business action using the current request identity."""

    from app.services.request_context import get_request_context

    context = get_request_context()
    if context.app_mode != "use":
        return None
    return AuditLogStore(db_path).record(
        event_type=event_type,
        app_mode=context.app_mode,
        user_id=context.user_id,
        request_id=context.request_id,
        action=action,
        status_code=status_code,
        metadata=metadata,
    )
