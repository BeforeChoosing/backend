"""Hybrid local retrieval with optional Bailian embedding and reranking."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from app.config import Settings, get_settings
from app.knowledge.retriever import KnowledgeChunk, KnowledgeRetriever
from app.knowledge.vector_index import LocalVectorIndex, VectorHit, chunk_embedding_text
from app.services.bailian_retrieval import (
    DashScopeEmbeddingGateway,
    DashScopeRerankGateway,
)
from app.services.model_response_cache import ModelResponseCache

logger = logging.getLogger(__name__)


class HybridKnowledgeRetriever:
    """Combine FTS5 recall, local cosine similarity and Bailian reranking.

    The class deliberately degrades to deterministic FTS5 ranking when the
    vector index is not built or a remote retrieval call is unavailable. This
    keeps the local Demo usable without fabricating model output.
    """

    def __init__(
        self,
        source_dir: str | Path,
        db_path: str | Path,
        *,
        settings: Settings | None = None,
        retriever: KnowledgeRetriever | None = None,
        embedding_gateway: Any | None = None,
        rerank_gateway: Any | None = None,
        cache: ModelResponseCache | None = None,
    ):
        self.settings = settings or get_settings()
        self.base = retriever or KnowledgeRetriever(source_dir, db_path)
        self.index = LocalVectorIndex(self.base.db_path)
        self.embedding_gateway = embedding_gateway or DashScopeEmbeddingGateway(
            self.settings
        )
        self.rerank_gateway = rerank_gateway or DashScopeRerankGateway(self.settings)
        self.cache = cache or ModelResponseCache(self.base.db_path)
        self.candidate_limit = max(1, min(int(getattr(self.settings, "rag_candidate_limit", 20)), 100))
        self.rerank_limit = max(1, min(int(getattr(self.settings, "rag_rerank_limit", 5)), 100))
        self.last_diagnostics: dict[str, Any] = {}

    def search(
        self,
        query: str,
        *,
        corpus: str,
        limit: int = 5,
        document_id: str | None = None,
    ) -> list[KnowledgeChunk]:
        normalized_query = str(query).strip()
        bounded_limit = max(1, min(int(limit), 100))
        candidate_limit = max(self.candidate_limit, bounded_limit)
        lexical = self.base.search(
            normalized_query,
            corpus=corpus,
            document_id=document_id,
            limit=candidate_limit,
        )
        lexical_by_id = {chunk.id: chunk for chunk in lexical}
        vector_hits: list[VectorHit] = []
        vector_by_id: dict[str, float] = {}
        vector_error = ""

        if self.index.ready:
            try:
                query_vector = self._query_vector(normalized_query)
                vector_hits = self.index.search(
                    query_vector,
                    corpus=corpus,
                    document_id=document_id,
                    model=getattr(self.settings, "bailian_embedding_model", None),
                    dimension=getattr(self.settings, "bailian_embedding_dimension", None),
                    source_fingerprint=self.base.source_fingerprint,
                    limit=candidate_limit,
                )
                vector_by_id = {hit.chunk_id: hit.score for hit in vector_hits}
            except Exception as exc:  # noqa: BLE001 - retrieval must retain FTS fallback
                vector_error = str(exc)
                logger.warning("local vector retrieval unavailable: %s", exc)

        candidate_ids = list(dict.fromkeys(
            [chunk.id for chunk in lexical]
            + [hit.chunk_id for hit in vector_hits]
        ))[:candidate_limit]
        candidates = self.base.get_chunks_by_ids(candidate_ids)
        if not candidates:
            self.last_diagnostics = {
                "mode": "none",
                "vector_used": bool(vector_hits),
                "rerank_used": False,
                "vector_error": vector_error,
            }
            return []

        # The expanded 26-query evaluation favors the local semantic vector
        # ranking. In vector mode, avoid an additional paid rerank request and
        # return the cached local cosine order, with lexical candidates only
        # filling a short result set when necessary.
        retriever_mode = str(getattr(self.settings, "rag_retriever_mode", "hybrid")).lower()
        if vector_hits and retriever_mode in {"vector", "semantic", "vector_only"}:
            by_id = {chunk.id: chunk for chunk in candidates}
            vector_ranked = [
                replace(by_id[hit.chunk_id], score=hit.score)
                for hit in vector_hits
                if hit.chunk_id in by_id
            ]
            seen_ids = {chunk.id for chunk in vector_ranked}
            if len(vector_ranked) < bounded_limit:
                vector_ranked.extend(
                    replace(chunk, score=chunk.score)
                    for chunk in lexical
                    if chunk.id not in seen_ids
                )
            self.last_diagnostics = {
                "mode": "vector",
                "vector_used": True,
                "rerank_used": False,
                "vector_error": vector_error,
            }
            return vector_ranked[:bounded_limit]

        reranked = self._try_rerank(normalized_query, candidates, bounded_limit)
        if reranked is not None:
            self.last_diagnostics = {
                "mode": "hybrid+rereank" if vector_hits else "fts+rereank",
                "vector_used": bool(vector_hits),
                "rerank_used": True,
                "vector_error": vector_error,
            }
            return reranked[:bounded_limit]

        fallback = self._combined_fallback(candidates, lexical_by_id, vector_by_id)
        self.last_diagnostics = {
            "mode": "hybrid" if vector_hits else "fts",
            "vector_used": bool(vector_hits),
            "rerank_used": False,
            "vector_error": vector_error,
        }
        return fallback[:bounded_limit]

    def _query_vector(self, query: str) -> list[float]:
        model = getattr(self.settings, "bailian_embedding_model", "")
        dimension = int(getattr(self.settings, "bailian_embedding_dimension", 0))
        cache_key = ModelResponseCache.fingerprint(
            {
                "model": model,
                "dimension": dimension,
                "source_fingerprint": self.base.source_fingerprint,
                "query": query,
            }
        )
        cached = self.cache.get("rag-query-embedding", cache_key)
        if cached is not None:
            vector = _read_vector(cached.get("embedding"), dimension)
            if vector:
                return vector

        vectors = self.embedding_gateway.embed([query], text_type="query")
        if len(vectors) != 1:
            raise ValueError("Embedding 查询向量数量不正确。")
        vector = _read_vector(vectors[0], dimension)
        if not vector:
            raise ValueError("Embedding 查询向量为空或维度不正确。")
        self.cache.set("rag-query-embedding", cache_key, {"embedding": vector})
        return vector

    def _try_rerank(
        self,
        query: str,
        candidates: list[KnowledgeChunk],
        limit: int,
    ) -> list[KnowledgeChunk] | None:
        if not candidates:
            return []
        top_n = min(len(candidates), max(limit, self.rerank_limit))
        candidate_signature = [
            {
                "id": chunk.id,
                "text": chunk_embedding_text(chunk),
            }
            for chunk in candidates
        ]
        cache_key = ModelResponseCache.fingerprint(
            {
                "model": getattr(self.settings, "bailian_rerank_model", ""),
                "query": query,
                "top_n": top_n,
                "source_fingerprint": self.base.source_fingerprint,
                "candidates": candidate_signature,
            }
        )
        hits = _read_rerank_hits(self.cache.get("rag-rerank", cache_key), len(candidates))
        if hits is None:
            try:
                response = self.rerank_gateway.rerank(
                    query,
                    [item["text"] for item in candidate_signature],
                    top_n=top_n,
                )
                hits = _read_rerank_hits(
                    {
                        "hits": [
                            {
                                "index": hit.index,
                                "score": hit.relevance_score,
                            }
                            for hit in response
                        ]
                    },
                    len(candidates),
                )
                if hits is None:
                    raise ValueError("Rerank 返回结果无法校验。")
                self.cache.set(
                    "rag-rerank",
                    cache_key,
                    {"hits": [{"index": index, "score": score} for index, score in hits]},
                )
            except Exception as exc:  # noqa: BLE001 - keep deterministic fallback
                logger.warning("Bailian rerank unavailable: %s", exc)
                return None

        by_index = {index: score for index, score in hits}
        result: list[KnowledgeChunk] = []
        for index, score in hits:
            if 0 <= index < len(candidates):
                result.append(replace(candidates[index], score=score))
        # If the remote endpoint returns fewer than requested, retain the
        # remaining local candidates after the scored prefix.
        if len(result) < limit:
            for index, chunk in enumerate(candidates):
                if index in by_index:
                    continue
                result.append(replace(chunk, score=-1.0))
                if len(result) >= limit:
                    break
        return result

    @staticmethod
    def _combined_fallback(
        candidates: list[KnowledgeChunk],
        lexical_by_id: dict[str, KnowledgeChunk],
        vector_by_id: dict[str, float],
    ) -> list[KnowledgeChunk]:
        lexical_max = max((chunk.score for chunk in lexical_by_id.values()), default=0.0)
        vector_max = max(vector_by_id.values(), default=0.0)
        ranked: list[KnowledgeChunk] = []
        for chunk in candidates:
            lexical_chunk = lexical_by_id.get(chunk.id)
            lexical_score = lexical_chunk.score if lexical_chunk else 0.0
            vector_score = vector_by_id.get(chunk.id, 0.0)
            lexical_norm = lexical_score / lexical_max if lexical_max > 0 else 0.0
            vector_norm = vector_score / vector_max if vector_max > 0 else 0.0
            if vector_by_id:
                combined = 0.6 * lexical_norm + 0.4 * vector_norm
            else:
                combined = lexical_norm
            ranked.append(replace(chunk, score=combined))
        ranked.sort(key=lambda chunk: (-chunk.score, chunk.id))
        return ranked


def _read_vector(value: Any, dimension: int) -> list[float]:
    if not isinstance(value, list) or not value:
        return []
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return []
    vector = [float(item) for item in value]
    if dimension > 0 and len(vector) != dimension:
        return []
    return vector


def _read_rerank_hits(
    payload: dict[str, Any] | None,
    candidate_count: int,
) -> list[tuple[int, float]] | None:
    if not isinstance(payload, dict):
        return None
    raw_hits = payload.get("hits")
    if not isinstance(raw_hits, list):
        return None
    result: list[tuple[int, float]] = []
    seen: set[int] = set()
    for item in raw_hits:
        if not isinstance(item, dict):
            return None
        index = item.get("index")
        score = item.get("score", item.get("relevance_score"))
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= candidate_count
            or index in seen
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
        ):
            return None
        seen.add(index)
        result.append((index, float(score)))
    if not result:
        return None
    result.sort(key=lambda item: (-item[1], item[0]))
    return result
