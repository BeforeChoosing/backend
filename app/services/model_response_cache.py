import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator


class ModelResponseCache:
    """Persist validated model responses for identical deterministic inputs."""

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
                CREATE TABLE IF NOT EXISTS model_response_cache (
                    namespace TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (namespace, cache_key)
                )
                """
            )

    @staticmethod
    def fingerprint(payload: Any) -> str:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, namespace: str, cache_key: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT response_json
                FROM model_response_cache
                WHERE namespace = ? AND cache_key = ?
                """,
                (namespace, cache_key),
            ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row["response_json"])
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def set(self, namespace: str, cache_key: str, response: dict[str, Any]) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO model_response_cache (
                    namespace, cache_key, response_json, created_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(namespace, cache_key) DO UPDATE SET
                    response_json = excluded.response_json,
                    created_at = excluded.created_at
                """,
                (
                    namespace,
                    cache_key,
                    json.dumps(response, ensure_ascii=False),
                    timestamp,
                ),
            )
