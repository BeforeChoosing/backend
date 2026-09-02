"""Append-only storage for formal-mode profile exploration turns."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4


class ConversationStore:
    """Keep the exact text conversation used to build candidate evidence.

    This table is separate from ``audit_events`` so audit metadata remains
    bounded while the authenticated owner can still trace the source of every
    candidate card. It is local-only and is never written in demo mode.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.db_path), timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS profile_conversation_turns (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    experience_text TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_profile_conversation_user_time
                ON profile_conversation_turns(user_id, created_at)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS profile_conversation_snapshots (
                    id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, id)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_profile_snapshot_user_time
                ON profile_conversation_snapshots(user_id, updated_at)
                """
            )

    def record_turn(
        self,
        *,
        user_id: str,
        request_id: str,
        trace_id: str,
        experience_text: str,
        messages: Sequence[Mapping[str, Any]],
        response: Mapping[str, Any],
    ) -> str:
        turn_id = f"profile-turn-{uuid4().hex}"
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO profile_conversation_turns (
                    id, user_id, request_id, trace_id, experience_text,
                    messages_json, response_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    user_id,
                    request_id,
                    trace_id,
                    experience_text,
                    json.dumps(list(messages), ensure_ascii=False, default=str),
                    json.dumps(dict(response), ensure_ascii=False, default=str),
                    created_at,
                ),
            )
        return turn_id

    def recent(self, *, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 500))
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id, user_id, request_id, trace_id, experience_text,
                       messages_json, response_json, created_at
                FROM profile_conversation_turns
                WHERE user_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (user_id, bounded_limit),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for key in ("messages_json", "response_json"):
                try:
                    item[key.removesuffix("_json")] = json.loads(item.pop(key))
                except (TypeError, json.JSONDecodeError):
                    item[key.removesuffix("_json")] = [] if key == "messages_json" else {}
            result.append(item)
        return result

    def upsert_snapshot(
        self,
        *,
        user_id: str,
        conversation_id: str,
        snapshot: Mapping[str, Any],
        max_per_user: int = 50,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        payload = dict(snapshot)
        title = str(payload.get("title") or "未命名对话")
        with self._connection() as connection:
            existing = connection.execute(
                """
                SELECT created_at
                FROM profile_conversation_snapshots
                WHERE user_id = ? AND id = ?
                """,
                (user_id, conversation_id),
            ).fetchone()
            created_at = str(existing["created_at"]) if existing else now
            connection.execute(
                """
                INSERT INTO profile_conversation_snapshots (
                    id, user_id, title, payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, id) DO UPDATE SET
                    title = excluded.title,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    conversation_id,
                    user_id,
                    title,
                    json.dumps(payload, ensure_ascii=False, default=str),
                    created_at,
                    now,
                ),
            )
            connection.execute(
                """
                DELETE FROM profile_conversation_snapshots
                WHERE user_id = ? AND id NOT IN (
                    SELECT id
                    FROM profile_conversation_snapshots
                    WHERE user_id = ?
                    ORDER BY updated_at DESC, id DESC
                    LIMIT ?
                )
                """,
                (user_id, user_id, max(1, min(int(max_per_user), 100))),
            )
        return {
            **payload,
            "id": conversation_id,
            "created_at": created_at,
            "updated_at": now,
        }

    def list_snapshots(self, *, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 50))
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id, payload_json, created_at, updated_at
                FROM profile_conversation_snapshots
                WHERE user_id = ?
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (user_id, bounded_limit),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            result.append(
                {
                    **payload,
                    "id": row["id"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            )
        return result

    def delete_snapshot(self, *, user_id: str, conversation_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                DELETE FROM profile_conversation_snapshots
                WHERE user_id = ? AND id = ?
                """,
                (user_id, conversation_id),
            )
        return cursor.rowcount > 0
