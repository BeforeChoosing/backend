from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Iterable


MAX_CHUNK_LENGTH = 1200
_ASCII_TERM_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]*")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]{2,}")
_STOP_TERMS = {
    "这个",
    "如何",
    "什么",
    "哪些",
    "以及",
    "可以",
    "能力",
    "用户",
    "岗位",
    "产品",
    "工作",
    "进行",
    "需要",
}


@dataclass(frozen=True)
class KnowledgeChunk:
    id: str
    document_id: str
    document_title: str
    corpus: str
    content: str
    heading_path: tuple[str, ...]
    source_locator: str
    trust_level: str
    source_note: str
    score: float


@dataclass(frozen=True)
class _DocumentMeta:
    path: str
    document_id: str
    corpus: str
    trust_level: str
    source_note: str


class KnowledgeRetriever:
    """Small deterministic local retriever for the extracted Markdown corpus.

    The corpus is intentionally small for the competition demo. SQLite FTS5 is
    built when available, while the final ranking also uses character n-grams so
    Chinese queries remain useful on systems without a Chinese tokenizer.
    """

    def __init__(self, source_dir: str | Path, db_path: str | Path):
        self.source_dir = Path(source_dir).expanduser()
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_diagnostics: dict[str, object] = {}
        self._ensure_index()

    def rebuild(self) -> int:
        documents = sorted(self.source_dir.rglob("*.md"))
        manifest = self._read_manifest()
        document_meta = self._document_meta(manifest)
        fingerprint = self._fingerprint(documents)
        with self._connection() as connection:
            connection.executescript(
                """
                DROP TABLE IF EXISTS knowledge_chunks_fts;
                DROP TABLE IF EXISTS knowledge_chunks;
                DROP TABLE IF EXISTS knowledge_documents;
                DROP TABLE IF EXISTS knowledge_embeddings;
                DROP TABLE IF EXISTS knowledge_meta;

                CREATE TABLE knowledge_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE knowledge_documents (
                    document_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    corpus TEXT NOT NULL,
                    trust_level TEXT NOT NULL,
                    source_note TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active'
                );

                CREATE TABLE knowledge_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    heading_path TEXT NOT NULL,
                    source_locator TEXT NOT NULL,
                    corpus TEXT NOT NULL,
                    trust_level TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active'
                );

                CREATE VIRTUAL TABLE knowledge_chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    document_title,
                    heading_path,
                    content
                );
                """
            )
            chunk_count = 0
            for path in documents:
                relative_path = path.relative_to(self.source_dir).as_posix()
                text = path.read_text(encoding="utf-8")
                meta = document_meta.get(relative_path) or self._fallback_meta(relative_path)
                title, chunks = _parse_markdown(text)
                content_hash = sha256(text.encode("utf-8")).hexdigest()
                connection.execute(
                    """
                    INSERT INTO knowledge_documents
                    (document_id, path, title, corpus, trust_level, source_note, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        meta.document_id,
                        relative_path,
                        title,
                        meta.corpus,
                        meta.trust_level,
                        meta.source_note,
                        content_hash,
                    ),
                )
                for chunk_index, (heading_path, content) in enumerate(chunks):
                    chunk_id = _chunk_id(meta.document_id, chunk_index, content)
                    locator = f"{relative_path}#{' > '.join(heading_path)}"
                    connection.execute(
                        """
                        INSERT INTO knowledge_chunks
                        (id, document_id, chunk_index, content, heading_path,
                         source_locator, corpus, trust_level)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chunk_id,
                            meta.document_id,
                            chunk_index,
                            content,
                            json.dumps(heading_path, ensure_ascii=False),
                            locator,
                            meta.corpus,
                            meta.trust_level,
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO knowledge_chunks_fts
                        (chunk_id, document_title, heading_path, content)
                        VALUES (?, ?, ?, ?)
                        """,
                        (chunk_id, title, " ".join(heading_path), content),
                    )
                    chunk_count += 1
            connection.execute(
                "INSERT INTO knowledge_meta (key, value) VALUES (?, ?)",
                ("source_fingerprint", fingerprint),
            )
            connection.execute(
                "INSERT INTO knowledge_meta (key, value) VALUES (?, ?)",
                ("chunk_count", str(chunk_count)),
            )
            return chunk_count

    @property
    def chunk_count(self) -> int:
        """Return the number of active indexed chunks."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value FROM knowledge_meta WHERE key = 'chunk_count'"
            ).fetchone()
        return int(row["value"]) if row else 0

    @property
    def source_fingerprint(self) -> str:
        """Return the fingerprint of the Markdown source used to build the index."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value FROM knowledge_meta WHERE key = 'source_fingerprint'"
            ).fetchone()
        return str(row["value"]) if row else ""

    def list_chunks(
        self,
        *,
        corpus: str | None = None,
        document_id: str | None = None,
    ) -> list[KnowledgeChunk]:
        """Materialize active chunks for local vector indexing or evaluation."""
        filters = ["c.status = 'active'"]
        params: list[str] = []
        if corpus:
            filters.append("c.corpus = ?")
            params.append(corpus)
        if document_id:
            filters.append("c.document_id = ?")
            params.append(document_id)
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT c.*, d.title AS document_title, d.source_note
                FROM knowledge_chunks c
                JOIN knowledge_documents d ON d.document_id = c.document_id
                WHERE {' AND '.join(filters)}
                ORDER BY c.document_id, c.chunk_index, c.id
                """,
                params,
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[KnowledgeChunk]:
        """Return active chunks in the same order as the supplied IDs."""
        if not chunk_ids:
            return []
        unique_ids = list(dict.fromkeys(chunk_ids))
        placeholders = ",".join("?" for _ in unique_ids)
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT c.*, d.title AS document_title, d.source_note
                FROM knowledge_chunks c
                JOIN knowledge_documents d ON d.document_id = c.document_id
                WHERE c.status = 'active' AND c.id IN ({placeholders})
                """,
                unique_ids,
            ).fetchall()
        by_id = {row["id"]: self._row_to_chunk(row) for row in rows}
        return [by_id[chunk_id] for chunk_id in unique_ids if chunk_id in by_id]

    def search(
        self,
        query: str,
        *,
        corpus: str,
        limit: int = 5,
        document_id: str | None = None,
    ) -> list[KnowledgeChunk]:
        terms = _query_terms(query)
        if not terms:
            return []
        with self._connection() as connection:
            candidates = self._candidate_ids(connection, terms)
            params: list[str] = [corpus]
            filters = ["c.corpus = ?", "c.status = 'active'"]
            if document_id:
                filters.append("c.document_id = ?")
                params.append(document_id)
            if candidates:
                placeholders = ",".join("?" for _ in candidates)
                filters.append(f"c.id IN ({placeholders})")
                params.extend(candidates)
            rows = connection.execute(
                f"""
                SELECT c.*, d.title AS document_title, d.source_note
                FROM knowledge_chunks c
                JOIN knowledge_documents d ON d.document_id = c.document_id
                WHERE {' AND '.join(filters)}
                """,
                params,
            ).fetchall()

        scored: list[KnowledgeChunk] = []
        for row in rows:
            haystack = " ".join(
                [row["document_title"], row["heading_path"], row["content"]]
            ).lower()
            matched = [term for term in terms if term in haystack]
            if not matched:
                continue
            score = _score_terms(query, terms, matched, haystack)
            scored.append(
                self._row_to_chunk(row, score=score)
            )
        scored.sort(key=lambda chunk: (-chunk.score, chunk.id))
        return scored[: max(1, min(limit, 100))]

    def search_many(
        self,
        queries: Iterable[str],
        *,
        corpus: str,
        limit: int = 5,
        document_id: str | None = None,
    ) -> list[KnowledgeChunk]:
        """Fuse several lexical intents without concatenating their text."""

        normalized_queries: list[str] = []
        seen: set[str] = set()
        for value in queries:
            normalized = " ".join(str(value).split()).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            normalized_queries.append(normalized)
        bounded_limit = max(1, min(int(limit), 100))
        if not normalized_queries:
            self.last_diagnostics = {
                "mode": "multi-query-empty",
                "query_count": 0,
                "vector_used": False,
                "rerank_used": False,
            }
            return []

        per_query: list[list[KnowledgeChunk]] = [
            self.search(
                query,
                corpus=corpus,
                document_id=document_id,
                limit=max(20, bounded_limit),
            )
            for query in normalized_queries
        ]
        by_id = {
            chunk.id: chunk
            for chunks in per_query
            for chunk in chunks
        }
        scores = {chunk_id: 0.0 for chunk_id in by_id}
        per_query_ids: list[list[str]] = []
        for query, chunks in zip(normalized_queries, per_query):
            per_query_ids.append([chunk.id for chunk in chunks])
            query_terms = set(_query_terms(query))
            for rank, chunk in enumerate(chunks, start=1):
                scores[chunk.id] += 1.0 / (60.0 + rank)
                if query_terms.intersection(_query_terms(" ".join(chunk.heading_path))):
                    scores[chunk.id] += 0.008

        ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
        result = [replace(by_id[chunk_id], score=scores[chunk_id]) for chunk_id in ordered]
        self.last_diagnostics = {
            "mode": "multi-query-fts",
            "query_count": len(normalized_queries),
            "vector_used": False,
            "rerank_used": False,
            "embedding_batch_calls": 0,
            "per_query_result_ids": per_query_ids,
            "query_coverage": round(
                sum(bool(ids) for ids in per_query_ids) / len(per_query_ids),
                4,
            ),
        }
        return result[:bounded_limit]

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row, *, score: float = 0.0) -> KnowledgeChunk:
        return KnowledgeChunk(
            id=row["id"],
            document_id=row["document_id"],
            document_title=row["document_title"],
            corpus=row["corpus"],
            content=row["content"],
            heading_path=tuple(json.loads(row["heading_path"])),
            source_locator=row["source_locator"],
            trust_level=row["trust_level"],
            source_note=row["source_note"],
            score=score,
        )

    def _ensure_index(self) -> None:
        documents = sorted(self.source_dir.rglob("*.md"))
        if not documents:
            raise FileNotFoundError(f"本地知识库目录为空：{self.source_dir}")
        fingerprint = self._fingerprint(documents)
        try:
            with self._connection() as connection:
                value = connection.execute(
                    "SELECT value FROM knowledge_meta WHERE key = 'source_fingerprint'"
                ).fetchone()
            if value and value["value"] == fingerprint:
                return
        except sqlite3.OperationalError:
            pass
        self.rebuild()

    def _candidate_ids(
        self,
        connection: sqlite3.Connection,
        terms: list[str],
    ) -> list[str]:
        ascii_terms = [term for term in terms if term.isascii() and len(term) > 1]
        if not ascii_terms:
            return []
        match = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in ascii_terms)
        try:
            rows = connection.execute(
                "SELECT chunk_id FROM knowledge_chunks_fts WHERE knowledge_chunks_fts MATCH ?",
                (match,),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [row["chunk_id"] for row in rows]

    def _read_manifest(self) -> dict:
        path = self.source_dir / "manifest.json"
        if not path.exists():
            # Keep the manifest beside the public corpus when the corpus is
            # mounted as ``knowledge/public``.
            path = self.source_dir.parent / "manifest.json"
        if not path.exists():
            return {"documents": []}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _document_meta(manifest: dict) -> dict[str, _DocumentMeta]:
        result: dict[str, _DocumentMeta] = {}
        for item in manifest.get("documents", []):
            if not isinstance(item, dict) or not item.get("path"):
                continue
            result[str(item["path"])] = _DocumentMeta(
                path=str(item["path"]),
                document_id=str(item.get("document_id") or item["path"]),
                corpus=str(item.get("corpus") or "career"),
                trust_level=str(item.get("trust_level") or "secondary_summary"),
                source_note=str(item.get("source_note") or ""),
            )
        return result

    @staticmethod
    def _fallback_meta(relative_path: str) -> _DocumentMeta:
        stem = Path(relative_path).with_suffix("").as_posix().replace("/", "-")
        corpus = "career" if relative_path.startswith("jobs/") else "capability_method"
        return _DocumentMeta(
            path=relative_path,
            document_id=f"doc-{stem}",
            corpus=corpus,
            trust_level="unreviewed",
            source_note="未提供 manifest 的本地资料。",
        )

    @staticmethod
    def _fingerprint(documents: Iterable[Path]) -> str:
        digest = sha256()
        for path in documents:
            digest.update(path.as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=10)
        connection.row_factory = sqlite3.Row
        return _ManagedConnection(connection)


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


def _chunk_id(document_id: str, index: int, content: str) -> str:
    value = sha256(f"{document_id}:{index}:{content}".encode("utf-8")).hexdigest()[:16]
    return f"chk-{value}"


def _parse_markdown(text: str) -> tuple[str, list[tuple[tuple[str, ...], str]]]:
    title = "未命名知识文档"
    headings: list[str] = []
    level_stack: list[tuple[int, str]] = []
    blocks: list[tuple[tuple[str, ...], str]] = []
    buffer: list[str] = []

    def flush() -> None:
        content = "\n".join(buffer).strip()
        buffer.clear()
        if not content:
            return
        current_path = tuple(item[1] for item in level_stack)
        for offset in range(0, len(content), MAX_CHUNK_LENGTH):
            blocks.append((current_path, content[offset : offset + MAX_CHUNK_LENGTH]))

    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            flush()
            level = len(match.group(1))
            heading = match.group(2).strip()
            if level == 1 and title == "未命名知识文档":
                title = heading
            level_stack = [item for item in level_stack if item[0] < level]
            level_stack.append((level, heading))
            headings = [item[1] for item in level_stack]
            continue
        if line.strip() == "" and buffer:
            flush()
            continue
        buffer.append(line)
    flush()
    return title, blocks


def _query_terms(query: str) -> list[str]:
    terms: list[str] = [item.lower() for item in _ASCII_TERM_RE.findall(query)]
    for run in _CJK_RUN_RE.findall(query):
        if run not in _STOP_TERMS:
            terms.append(run)
        terms.extend(
            run[index : index + 2]
            for index in range(len(run) - 1)
            if run[index : index + 2] not in _STOP_TERMS
        )
    deduplicated: list[str] = []
    for term in terms:
        if len(term) < 2 or term in deduplicated:
            continue
        deduplicated.append(term)
    return deduplicated[:80]


def _score_terms(query: str, terms: list[str], matched: list[str], haystack: str) -> float:
    score = sum(1.0 if len(term) <= 2 else 1.5 for term in matched)
    coverage = len(matched) / max(1, len(terms))
    score *= 0.7 + coverage
    if query.strip().lower() in haystack:
        score += 2.0
    return round(score, 4)
