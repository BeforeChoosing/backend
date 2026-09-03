import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from app.schemas.task_catalog import (
    DynamicTrialAnswer,
    DynamicTrialCoachUsage,
    DynamicTrialSession,
)
from app.schemas.trial import ObservedEvidence, TrialEvaluation


class DynamicTrialStore:
    """Persist generic sessions for the fixed 12-task workbench."""

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
                CREATE TABLE IF NOT EXISTS dynamic_trial_sessions (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    event_revealed INTEGER NOT NULL DEFAULT 0,
                    answer_json TEXT NOT NULL,
                    observed_evidence_json TEXT,
                    evaluation_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    submitted_at TEXT
                )
                """
            )

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _session_from_row(row: sqlite3.Row) -> DynamicTrialSession:
        return DynamicTrialSession.model_validate(
            {
                "id": row["id"],
                "task_id": row["task_id"],
                "status": row["status"],
                "event_revealed": bool(row["event_revealed"]),
                "answer": json.loads(row["answer_json"]),
                "created_at": datetime.fromisoformat(row["created_at"]),
                "updated_at": datetime.fromisoformat(row["updated_at"]),
                "submitted_at": datetime.fromisoformat(row["submitted_at"]) if row["submitted_at"] else None,
                "observed_evidence": json.loads(row["observed_evidence_json"]) if row["observed_evidence_json"] else None,
                "evaluation": json.loads(row["evaluation_json"]) if row["evaluation_json"] else None,
            }
        )

    @staticmethod
    def _get_row(connection: sqlite3.Connection, session_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM dynamic_trial_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise KeyError(session_id)
        return row

    def create_session(
        self,
        task_id: str,
        answer: DynamicTrialAnswer | None = None,
    ) -> DynamicTrialSession:
        session_id = f"dynamic-trial-{uuid4().hex}"
        timestamp = self._now()
        answer = answer or DynamicTrialAnswer()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO dynamic_trial_sessions (
                    id, task_id, status, event_revealed, answer_json, created_at, updated_at
                ) VALUES (?, ?, 'in_progress', 0, ?, ?, ?)
                """,
                (
                    session_id,
                    task_id,
                    json.dumps(answer.model_dump(mode="json"), ensure_ascii=False),
                    timestamp.isoformat(),
                    timestamp.isoformat(),
                ),
            )
            return self._session_from_row(self._get_row(connection, session_id))

    def get_session(self, session_id: str) -> DynamicTrialSession:
        with self._connection() as connection:
            return self._session_from_row(self._get_row(connection, session_id))

    def save_answer(self, session_id: str, answer: DynamicTrialAnswer) -> DynamicTrialSession:
        timestamp = self._now()
        with self._connection() as connection:
            row = self._get_row(connection, session_id)
            if row["status"] == "submitted":
                raise ValueError("已提交的试路会话不能继续修改。")
            persisted_answer = DynamicTrialAnswer.model_validate_json(row["answer_json"])
            answer.coach_usage = persisted_answer.coach_usage
            connection.execute(
                "UPDATE dynamic_trial_sessions SET answer_json = ?, updated_at = ? WHERE id = ?",
                (
                    json.dumps(answer.model_dump(mode="json"), ensure_ascii=False),
                    timestamp.isoformat(),
                    session_id,
                ),
            )
            return self._session_from_row(self._get_row(connection, session_id))

    def reveal_event(self, session_id: str) -> DynamicTrialSession:
        timestamp = self._now()
        with self._connection() as connection:
            row = self._get_row(connection, session_id)
            if row["status"] == "submitted":
                return self._session_from_row(row)
            connection.execute(
                "UPDATE dynamic_trial_sessions SET event_revealed = 1, updated_at = ? WHERE id = ?",
                (timestamp.isoformat(), session_id),
            )
            return self._session_from_row(self._get_row(connection, session_id))

    def record_coach_usage(
        self,
        session_id: str,
        usage: DynamicTrialCoachUsage,
    ) -> DynamicTrialSession:
        timestamp = self._now()
        with self._connection() as connection:
            row = self._get_row(connection, session_id)
            if row["status"] == "submitted":
                raise ValueError("已提交的试路会话不能继续使用 Coach。")
            answer = DynamicTrialAnswer.model_validate_json(row["answer_json"])
            answer.coach_usage.append(usage)
            connection.execute(
                "UPDATE dynamic_trial_sessions SET answer_json = ?, updated_at = ? WHERE id = ?",
                (
                    json.dumps(answer.model_dump(mode="json"), ensure_ascii=False),
                    timestamp.isoformat(),
                    session_id,
                ),
            )
            return self._session_from_row(self._get_row(connection, session_id))

    def submit(
        self,
        session_id: str,
        observed_evidence: ObservedEvidence,
        evaluation: TrialEvaluation,
    ) -> DynamicTrialSession:
        timestamp = self._now()
        with self._connection() as connection:
            row = self._get_row(connection, session_id)
            if row["status"] == "submitted":
                return self._session_from_row(row)
            connection.execute(
                """
                UPDATE dynamic_trial_sessions
                SET status = 'submitted', observed_evidence_json = ?, evaluation_json = ?,
                    submitted_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    json.dumps(observed_evidence.model_dump(mode="json"), ensure_ascii=False),
                    json.dumps(evaluation.model_dump(mode="json"), ensure_ascii=False),
                    timestamp.isoformat(),
                    timestamp.isoformat(),
                    session_id,
                ),
            )
            return self._session_from_row(self._get_row(connection, session_id))
