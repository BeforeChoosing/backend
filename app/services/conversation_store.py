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
