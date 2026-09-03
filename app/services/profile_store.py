import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from app.schemas.profile import (
    CardProposal,
    ProfileCard,
    ProfileCardPatchRequest,
    ProfileCardsResponse,
    ProfileEvidenceRecord,
    ProfileOverviewResponse,
)
from app.schemas.trial import ObservedEvidence, TrialEvaluation


class ProfileStore:
    """Persist the single local demo profile in a small SQLite database."""

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
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS profile_cards (
                    id TEXT PRIMARY KEY,
                    card_json TEXT NOT NULL,
                    source_trace_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS profile_versions (
                    version INTEGER PRIMARY KEY AUTOINCREMENT,
                    reason TEXT NOT NULL,
                    trace_id TEXT,
                    snapshot_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS profile_evidence (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL UNIQUE,
                    task_id TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    evaluation_json TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(profile_evidence)").fetchall()
            }
            if "evaluation_json" not in columns:
                connection.execute(
                    "ALTER TABLE profile_evidence ADD COLUMN evaluation_json TEXT"
                )

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _card_from_row(row: sqlite3.Row) -> ProfileCard:
        payload = json.loads(row["card_json"])
        return ProfileCard.model_validate(payload)

    @classmethod
    def _cards_from_connection(cls, connection: sqlite3.Connection) -> list[ProfileCard]:
        rows = connection.execute(
            "SELECT card_json FROM profile_cards ORDER BY created_at ASC, id ASC"
        ).fetchall()
        return [cls._card_from_row(row) for row in rows]

    @staticmethod
    def _snapshot(cards: list[ProfileCard]) -> str:
        return json.dumps(
            [card.model_dump(mode="json") for card in cards],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _response(
        connection: sqlite3.Connection,
        cards: list[ProfileCard] | None = None,
    ) -> ProfileCardsResponse:
        persisted_cards = cards if cards is not None else ProfileStore._cards_from_connection(connection)
        latest = connection.execute(
            "SELECT version, created_at FROM profile_versions ORDER BY version DESC LIMIT 1"
        ).fetchone()
        return ProfileCardsResponse(
            version=int(latest["version"]) if latest else 0,
            updated_at=(datetime.fromisoformat(latest["created_at"]) if latest else None),
            cards=persisted_cards,
        )

    def get_profile(self) -> ProfileCardsResponse:
        with self._connection() as connection:
            return self._response(connection)

    @staticmethod
    def _evidence_from_row(row: sqlite3.Row) -> ProfileEvidenceRecord:
        return ProfileEvidenceRecord(
            session_id=str(row["session_id"]),
            task_id=str(row["task_id"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            observed_evidence=ObservedEvidence.model_validate(json.loads(row["evidence_json"])),
            evaluation=(
                TrialEvaluation.model_validate(json.loads(row["evaluation_json"]))
                if row["evaluation_json"]
                else None
            ),
        )

    def get_evidence_records(self) -> list[ProfileEvidenceRecord]:
        """Return submitted task evidence in reverse chronological order."""
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT session_id, task_id, evidence_json, evaluation_json, created_at
                FROM profile_evidence
                ORDER BY created_at DESC, session_id DESC
                """
            ).fetchall()
            return [self._evidence_from_row(row) for row in rows]

    def get_profile_overview(self) -> ProfileOverviewResponse:
        with self._connection() as connection:
            cards = self._cards_from_connection(connection)
            latest = connection.execute(
                "SELECT version, created_at FROM profile_versions ORDER BY version DESC LIMIT 1"
            ).fetchone()
            evidence_rows = connection.execute(
                """
                SELECT session_id, task_id, evidence_json, evaluation_json, created_at
                FROM profile_evidence
                ORDER BY created_at DESC, session_id DESC
                """
            ).fetchall()
            evidence = [self._evidence_from_row(row) for row in evidence_rows]
            completed = list(dict.fromkeys(record.task_id for record in evidence))
            return ProfileOverviewResponse(
                version=int(latest["version"]) if latest else 0,
                updated_at=(datetime.fromisoformat(latest["created_at"]) if latest else None),
                cards=cards,
                evidence=evidence,
                completed_task_ids=completed,
            )

    def get_cards_by_ids(self, card_ids: list[str]) -> list[ProfileCard]:
        """Return only confirmed cards selected for the current career journey."""
        if not card_ids:
            return []
        placeholders = ",".join("?" for _ in card_ids)
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT card_json
                FROM profile_cards
                WHERE id IN ({placeholders})
                ORDER BY created_at ASC, id ASC
                """,
                card_ids,
            ).fetchall()
            return [self._card_from_row(row) for row in rows]

    def get_completed_task_ids(self) -> list[str]:
        """Return task IDs with submitted observed evidence for the local profile."""
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT task_id
                FROM profile_evidence
                GROUP BY task_id
                ORDER BY MIN(created_at) ASC, task_id ASC
                """
            ).fetchall()
            return [str(row["task_id"]) for row in rows]

    def _record_version(
        self,
        connection: sqlite3.Connection,
        *,
        reason: str,
        trace_id: str | None,
        cards: list[ProfileCard],
        timestamp: datetime,
    ) -> None:
        connection.execute(
            """
            INSERT INTO profile_versions (reason, trace_id, snapshot_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                reason,
                trace_id,
                self._snapshot(cards),
                timestamp.isoformat(),
            ),
        )

    def confirm_cards(
        self,
        cards: list[CardProposal],
        trace_id: str | None = None,
    ) -> ProfileCardsResponse:
        timestamp = self._now()
        with self._connection() as connection:
            for proposal in cards:
                proposal = CardProposal.model_validate(proposal.model_dump())
                merge_target_id = proposal.merge_target_card_id if proposal.resolution == "merge" else None
                if merge_target_id is None:
                    normalized_title = re.sub(r"[\s，。！？、,:：；;.!?]", "", proposal.title)
                    for candidate in self._cards_from_connection(connection):
                        candidate_title = re.sub(r"[\s，。！？、,:：；;.!?]", "", candidate.title)
                        if candidate_title == normalized_title:
                            merge_target_id = candidate.id
                            break
                merge_target = None
                if merge_target_id:
                    row = connection.execute(
                        "SELECT card_json FROM profile_cards WHERE id = ?",
                        (merge_target_id,),
                    ).fetchone()
                    if row is not None:
                        merge_target = self._card_from_row(row)

                if merge_target is not None:
                    history = [*merge_target.evidence_history, *proposal.evidence_history]
                    unique_history = []
                    seen_evidence: set[tuple[str, str]] = set()
                    for evidence in history:
                        key = (evidence.experience_id, evidence.evidence_quote)
                        if key not in seen_evidence:
                            unique_history.append(evidence)
                            seen_evidence.add(key)
                    detail = merge_target.detail
                    if proposal.detail not in detail:
                        detail = f"{detail}\n{proposal.detail}"[:600]
                    proposal = CardProposal.model_validate({
                        **merge_target.model_dump(exclude={"status", "source_trace_id", "created_at", "updated_at"}),
                        "id": merge_target.id,
                        "detail": detail,
                        "evidence_quote": proposal.evidence_quote,
                        "evidence_type": proposal.evidence_type,
                        "claim_level": proposal.claim_level,
                        "source_refs": list(dict.fromkeys([
                            *merge_target.source_refs,
                            *proposal.source_refs,
                        ]))[:10],
                        "experience_id": proposal.experience_id,
                        "evidence_history": [item.model_dump() for item in unique_history][-30:],
                        "resolution": "new",
                        "merge_target_card_id": None,
                    })
                existing = connection.execute(
                    "SELECT created_at, source_trace_id FROM profile_cards WHERE id = ?",
                    (proposal.id,),
                ).fetchone()
                created_at = (
                    datetime.fromisoformat(existing["created_at"])
                    if existing
                    else timestamp
                )
                source_trace_id = trace_id or (existing["source_trace_id"] if existing else None)
                profile_card = ProfileCard.model_validate(
                    {
                        **proposal.model_dump(),
                        "status": "confirmed",
                        "source_trace_id": source_trace_id,
                        "created_at": created_at,
                        "updated_at": timestamp,
                    }
                )
                connection.execute(
                    """
                    INSERT INTO profile_cards (
                        id, card_json, source_trace_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        card_json = excluded.card_json,
                        source_trace_id = excluded.source_trace_id,
                        updated_at = excluded.updated_at
                    """,
                    (
                        profile_card.id,
                        json.dumps(profile_card.model_dump(mode="json"), ensure_ascii=False),
                        profile_card.source_trace_id,
                        profile_card.created_at.isoformat(),
                        profile_card.updated_at.isoformat(),
                    ),
                )

            persisted_cards = self._cards_from_connection(connection)
            self._record_version(
                connection,
                reason="confirm",
                trace_id=trace_id,
                cards=persisted_cards,
                timestamp=timestamp,
            )
            return self._response(connection, persisted_cards)

    def update_card(
        self,
        card_id: str,
        patch: ProfileCardPatchRequest,
        trace_id: str | None = None,
    ) -> ProfileCardsResponse:
        timestamp = self._now()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT card_json FROM profile_cards WHERE id = ?", (card_id,)
            ).fetchone()
            if row is None:
                raise KeyError(card_id)

            current = self._card_from_row(row)
            updated = ProfileCard.model_validate(
                {
                    **current.model_dump(),
                    **patch.model_dump(exclude_unset=True),
                    "updated_at": timestamp,
                }
            )
            connection.execute(
                """
                UPDATE profile_cards
                SET card_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    json.dumps(updated.model_dump(mode="json"), ensure_ascii=False),
                    updated.updated_at.isoformat(),
                    card_id,
                ),
            )
            persisted_cards = self._cards_from_connection(connection)
            self._record_version(
                connection,
                reason="update",
                trace_id=trace_id,
                cards=persisted_cards,
                timestamp=timestamp,
            )
            return self._response(connection, persisted_cards)

    def delete_card(
        self,
        card_id: str,
        trace_id: str | None = None,
    ) -> ProfileCardsResponse:
        timestamp = self._now()
        with self._connection() as connection:
            deleted = connection.execute(
                "DELETE FROM profile_cards WHERE id = ?", (card_id,)
            )
            if deleted.rowcount == 0:
                raise KeyError(card_id)

            persisted_cards = self._cards_from_connection(connection)
            self._record_version(
                connection,
                reason="delete",
                trace_id=trace_id,
                cards=persisted_cards,
                timestamp=timestamp,
            )
            return self._response(connection, persisted_cards)

    def record_observed_evidence(
        self,
        session_id: str,
        evidence: ObservedEvidence,
        evaluation: TrialEvaluation | None = None,
    ) -> ProfileCardsResponse:
        """Write task evidence into the same local profile version stream."""
        timestamp = self._now()
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT 1 FROM profile_evidence WHERE session_id = ?", (session_id,)
            ).fetchone()
            if existing:
                if evaluation is not None:
                    connection.execute(
                        "UPDATE profile_evidence SET evaluation_json = ? WHERE session_id = ?",
                        (
                            json.dumps(evaluation.model_dump(mode="json"), ensure_ascii=False),
                            session_id,
                        ),
                    )
                return self._response(connection)

            connection.execute(
                """
                INSERT INTO profile_evidence (
                    id, session_id, task_id, evidence_json, evaluation_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    f"evidence-{session_id}",
                    session_id,
                    evidence.task_id,
                    json.dumps(evidence.model_dump(mode="json"), ensure_ascii=False),
                    json.dumps(evaluation.model_dump(mode="json"), ensure_ascii=False)
                    if evaluation is not None
                    else None,
                    timestamp.isoformat(),
                ),
            )
            persisted_cards = self._cards_from_connection(connection)
            self._record_version(
                connection,
                reason="trial_evidence",
                trace_id=session_id,
                cards=persisted_cards,
                timestamp=timestamp,
            )
            return self._response(connection, persisted_cards)
