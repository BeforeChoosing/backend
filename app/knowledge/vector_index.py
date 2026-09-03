"""Local SQLite vector index backed by Bailian-generated embeddings."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

from app.knowledge.retriever import KnowledgeChunk, KnowledgeRetriever


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    score: float


@dataclass(frozen=True)
class VectorBuildReport:
    total: int
    embedded: int
    reused: int
    removed: int


class LocalVectorIndex:
    """Store and search dense vectors locally, with no external vector DB."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @property
    def count(self) -> int:
        with self._connection() as connection:
            try:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM knowledge_embeddings"
                ).fetchone()
            except sqlite3.OperationalError:
                return 0
        return int(row["count"]) if row else 0

    @property
    def ready(self) -> bool:
        return self.count > 0

    def existing(self) -> dict[str, dict[str, Any]]:
        with self._connection() as connection:
            try:
                rows = connection.execute(
                    """
                    SELECT chunk_id, model, dimension, content_hash,
                           source_fingerprint
                    FROM knowledge_embeddings
                    """
                ).fetchall()
            except sqlite3.OperationalError:
                return {}
        return {row["chunk_id"]: dict(row) for row in rows}

    def upsert(
        self,
        entries: Sequence[tuple[str, str, int, str, str, Sequence[float]]],
    ) -> None:
        if not entries:
            return
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.executemany(
                """
                INSERT INTO knowledge_embeddings (
                    chunk_id, model, dimension, content_hash,
                    source_fingerprint, vector_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    model = excluded.model,
                    dimension = excluded.dimension,
                    content_hash = excluded.content_hash,
                    source_fingerprint = excluded.source_fingerprint,
                    vector_json = excluded.vector_json,
                    created_at = excluded.created_at
                """,
                [
                    (
                        chunk_id,
                        model,
                        dimension,
                        content_hash,
                        source_fingerprint,
                        json.dumps(list(vector), ensure_ascii=False),
                        timestamp,
                    )
                    for chunk_id, model, dimension, content_hash, source_fingerprint, vector in entries
                ],
            )

    def update_source_fingerprint(
        self,
        chunk_ids: set[str],
        source_fingerprint: str,
    ) -> None:
        """Refresh provenance for reused vectors without re-embedding them."""

        if not chunk_ids:
            return
        with self._connection() as connection:
            connection.executemany(
                """
                UPDATE knowledge_embeddings
                SET source_fingerprint = ?
                WHERE chunk_id = ?
                """,
                [(source_fingerprint, chunk_id) for chunk_id in chunk_ids],
            )

    def remove_except(
        self,
        chunk_ids: set[str],
        *,
        corpus: str | None = None,
        document_id: str | None = None,
    ) -> int:
        with self._connection() as connection:
            try:
                filters = ["1 = 1"]
                params: list[str] = []
                if corpus:
                    filters.append("c.corpus = ?")
                    params.append(corpus)
                if document_id:
                    filters.append("c.document_id = ?")
                    params.append(document_id)
                rows = connection.execute(
                    f"""
                    SELECT e.chunk_id
                    FROM knowledge_embeddings e
                    JOIN knowledge_chunks c ON c.id = e.chunk_id
                    WHERE {' AND '.join(filters)}
                    """,
                    params,
                ).fetchall()
            except sqlite3.OperationalError:
                return 0
            stale = [row["chunk_id"] for row in rows if row["chunk_id"] not in chunk_ids]
            if stale:
                connection.executemany(
                    "DELETE FROM knowledge_embeddings WHERE chunk_id = ?",
                    [(chunk_id,) for chunk_id in stale],
                )
        return len(stale)

    def search(
        self,
        query_vector: Sequence[float],
        *,
        corpus: str | None = None,
        document_id: str | None = None,
        model: str | None = None,
        dimension: int | None = None,
        source_fingerprint: str | None = None,
        limit: int = 20,
    ) -> list[VectorHit]:
        vector = _validated_vector(query_vector)
        if not vector:
            return []
        query_norm = math.sqrt(sum(value * value for value in vector))
        if query_norm <= 0:
            return []

        filters = ["c.status = 'active'"]
        params: list[str] = []
        if corpus:
            filters.append("c.corpus = ?")
            params.append(corpus)
        if document_id:
            filters.append("c.document_id = ?")
            params.append(document_id)
        if model:
            filters.append("e.model = ?")
            params.append(model)
        if dimension is not None:
            filters.append("e.dimension = ?")
            params.append(str(int(dimension)))
        if source_fingerprint:
            filters.append("e.source_fingerprint = ?")
            params.append(source_fingerprint)
        with self._connection() as connection:
            try:
                rows = connection.execute(
                    f"""
                    SELECT e.chunk_id, e.vector_json
                    FROM knowledge_embeddings e
                    JOIN knowledge_chunks c ON c.id = e.chunk_id
                    WHERE {' AND '.join(filters)}
                    """,
                    params,
                ).fetchall()
            except sqlite3.OperationalError:
                return []

        hits: list[VectorHit] = []
        for row in rows:
            try:
                candidate = _validated_vector(json.loads(row["vector_json"]))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if len(candidate) != len(vector):
                continue
            candidate_norm = math.sqrt(sum(value * value for value in candidate))
            if candidate_norm <= 0:
                continue
            score = sum(left * right for left, right in zip(vector, candidate)) / (
                query_norm * candidate_norm
            )
            hits.append(VectorHit(chunk_id=row["chunk_id"], score=score))
        hits.sort(key=lambda hit: (-hit.score, hit.chunk_id))
        return hits[: max(1, min(int(limit), 100))]

    def _ensure_schema(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    source_fingerprint TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def _connection(self) -> "_ManagedConnection":
        connection = sqlite3.connect(str(self.db_path), timeout=10)
        connection.row_factory = sqlite3.Row
        return _ManagedConnection(connection)


class VectorIndexBuilder:
    """Build only missing or changed vectors through the configured gateway."""

    def __init__(
        self,
        retriever: KnowledgeRetriever,
        gateway: Any,
        *,
        model: str,
        dimension: int,
        batch_size: int = 20,
    ):
        self.retriever = retriever
        self.gateway = gateway
        self.model = model
        self.dimension = int(dimension)
        self.batch_size = max(1, min(int(batch_size), 20))
        self.index = LocalVectorIndex(retriever.db_path)

    def build(
        self,
        *,
        corpus: str | None = None,
        document_id: str | None = None,
    ) -> VectorBuildReport:
        chunks = self.retriever.list_chunks(corpus=corpus, document_id=document_id)
        source_fingerprint = self.retriever.source_fingerprint
        existing = self.index.existing()
        pending: list[tuple[KnowledgeChunk, str, str]] = []
        reused = 0
        reused_ids: set[str] = set()
        valid_ids = {chunk.id for chunk in chunks}
        for chunk in chunks:
            text = chunk_embedding_text(chunk)
            content_hash = sha256(text.encode("utf-8")).hexdigest()
            row = existing.get(chunk.id)
            if (
                row
                and row["model"] == self.model
                and int(row["dimension"]) == self.dimension
                and row["content_hash"] == content_hash
            ):
                reused += 1
                reused_ids.add(chunk.id)
                continue
            pending.append((chunk, text, content_hash))

        embedded = 0
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            vectors = self.gateway.embed_many(
                [item[1] for item in batch],
                text_type="document",
            )
            if len(vectors) != len(batch):
                raise ValueError("Embedding 返回数量与待索引片段数量不一致。")
            entries = []
            for (chunk, _text, content_hash), vector in zip(batch, vectors):
                validated = _validated_vector(vector)
                if len(validated) != self.dimension:
                    raise ValueError("Embedding 返回维度与配置不一致。")
                entries.append(
                    (
                        chunk.id,
                        self.model,
                        self.dimension,
                        content_hash,
                        source_fingerprint,
                        validated,
                    )
                )
            self.index.upsert(entries)
            embedded += len(entries)

        # A source change invalidates the corpus fingerprint, but not a vector
        # whose embedding text is byte-for-byte unchanged. Refresh the stamp so
        # vector retrieval can keep its source guard while avoiding an API call.
        self.index.update_source_fingerprint(reused_ids, source_fingerprint)

        removed = self.index.remove_except(
            valid_ids,
            corpus=corpus,
            document_id=document_id,
        )
        return VectorBuildReport(
            total=len(chunks),
            embedded=embedded,
            reused=reused,
            removed=removed,
        )


def chunk_embedding_text(chunk: KnowledgeChunk) -> str:
    heading = " > ".join(chunk.heading_path)
    return "\n".join(
        item for item in (chunk.document_title, heading, chunk.content) if item
    )


def _validated_vector(values: Sequence[float] | Any) -> list[float]:
    if not isinstance(values, (list, tuple)) or not values:
        return []
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("向量包含非数字值。")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("向量包含非有限值。")
        result.append(numeric)
    return result


class _ManagedConnection:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def __enter__(self) -> sqlite3.Connection:
        return self.connection

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if exc_type:
                self.connection.rollback()
            else:
                self.connection.commit()
        finally:
            self.connection.close()
