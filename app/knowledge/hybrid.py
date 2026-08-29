"""Hybrid local retrieval with optional Bailian embedding and reranking."""

from __future__ import annotations

import logging
import re
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
        # Adaptive mode only spends a rerank request when the local semantic
        # ranking is ambiguous.  The defaults are intentionally conservative:
        # most queries stay on the cached local vector path.
        self.adaptive_margin = max(
            0.0, float(getattr(self.settings, "rag_adaptive_margin", 0.055))
        )
        self.adaptive_rerank_min_margin = max(
            0.0,
            float(
                getattr(
                    self.settings,
                    "rag_adaptive_rerank_min_margin",
                    0.02,
                )
            ),
        )
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

        candidate_ids = self._merge_candidate_ids(
            lexical,
            vector_hits,
            limit=candidate_limit,
        )
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
            vector_ranked = self._vector_ranked(
                vector_hits,
                candidates,
                lexical,
                limit=bounded_limit,
            )
            self.last_diagnostics = {
                "mode": "vector",
                "vector_used": True,
                "rerank_used": False,
                "adaptive_rerank_triggered": False,
                "vector_error": vector_error,
            }
            return vector_ranked[:bounded_limit]

        if vector_hits and retriever_mode in {"adaptive", "adaptive_rerank"}:
            vector_ranked = self._vector_ranked(
                vector_hits,
                candidates,
                lexical,
                limit=bounded_limit,
            )
            margin = self._vector_margin(vector_hits)
            should_rerank = margin < self.adaptive_margin
            if not should_rerank:
                self.last_diagnostics = {
                    "mode": "adaptive-vector",
                    "vector_used": True,
                    "rerank_used": False,
                    "adaptive_rerank_triggered": False,
                    "adaptive_margin": margin,
                    "adaptive_threshold": self.adaptive_margin,
                    "vector_error": vector_error,
                }
                return vector_ranked[:bounded_limit]

            reranked = self._try_rerank(normalized_query, candidates, bounded_limit)
            rerank_margin = self._rerank_margin(reranked)
            if reranked is not None and rerank_margin >= self.adaptive_rerank_min_margin:
                # A remote cross-encoder is a secondary signal.  Pin the
                # local semantic winner so an unstable reranker cannot turn a
                # previously correct top-1 result into a miss.  It may still
                # reorder the remaining candidates and improve MRR/Hit@K.
                vector_top = vector_ranked[0] if vector_ranked else None
                if vector_top is not None:
                    vector_ids = {chunk.id for chunk in vector_ranked}
                    reranked_tail = [
                        chunk
                        for chunk in reranked
                        if chunk.id != vector_top.id and chunk.id in vector_ids
                    ]
                    # Keep the requested-K vector candidate set intact.  The
                    # remote model can reorder those candidates, but cannot
                    # replace them with out-of-set documents and reduce recall.
                    reranked = [vector_top] + reranked_tail
                    seen = {chunk.id for chunk in reranked}
                    reranked.extend(
                        chunk for chunk in vector_ranked if chunk.id not in seen
                    )
                self.last_diagnostics = {
                    "mode": "adaptive-rerank",
                    "vector_used": True,
                    "rerank_used": True,
                    "adaptive_rerank_triggered": True,
                    "adaptive_vector_top_pinned": vector_top is not None,
                    "adaptive_margin": margin,
                    "adaptive_threshold": self.adaptive_margin,
                    "rerank_margin": rerank_margin,
                    "vector_error": vector_error,
                }
                return reranked[:bounded_limit]

            self.last_diagnostics = {
                "mode": "adaptive-vector-fallback",
                "vector_used": True,
                "rerank_used": False,
                "adaptive_rerank_triggered": True,
                "adaptive_margin": margin,
                "adaptive_threshold": self.adaptive_margin,
                "rerank_margin": rerank_margin,
                "vector_error": vector_error,
            }
            return vector_ranked[:bounded_limit]

        reranked = self._try_rerank(normalized_query, candidates, bounded_limit)
        if reranked is not None:
            self.last_diagnostics = {
                "mode": "hybrid+rereank" if vector_hits else "fts+rereank",
                "vector_used": bool(vector_hits),
                "rerank_used": True,
                "adaptive_rerank_triggered": False,
                "vector_error": vector_error,
            }
            return reranked[:bounded_limit]

        fallback = self._combined_fallback(candidates, lexical_by_id, vector_by_id)
        self.last_diagnostics = {
            "mode": "hybrid" if vector_hits else "fts",
            "vector_used": bool(vector_hits),
            "rerank_used": False,
            "adaptive_rerank_triggered": False,
            "vector_error": vector_error,
        }
        return fallback[:bounded_limit]

    def search_many(
        self,
        queries: Sequence[str],
        *,
        corpus: str,
        limit: int = 5,
        document_id: str | None = None,
    ) -> list[KnowledgeChunk]:
        """Retrieve several focused intents with one batched embedding call.

        Career recommendations are grounded by a role query and one query per
        confirmed ability card.  Running those intents independently and
        fusing their ranked candidates avoids the dilution caused by one long
        concatenated query.  Query vectors are cached individually, while
        missing vectors are sent through ``embed_many`` in one API request.
        The final local MMR pass limits near-duplicate chunks so the citation
        budget covers more distinct sections of the local knowledge base.
        """

        normalized_queries: list[str] = []
        seen_queries: set[str] = set()
        for value in queries:
            normalized = " ".join(str(value).split()).strip()
            if not normalized or normalized in seen_queries:
                continue
            seen_queries.add(normalized)
            normalized_queries.append(normalized)
        bounded_limit = max(1, min(int(limit), 100))
        candidate_limit = max(self.candidate_limit, bounded_limit)
        if not normalized_queries:
            self.last_diagnostics = {
                "mode": "multi-query-empty",
                "query_count": 0,
                "vector_used": False,
                "rerank_used": False,
            }
            return []

        lexical_lists: list[list[KnowledgeChunk]] = []
        for query in normalized_queries:
            lexical_lists.append(
                self.base.search(
                    query,
                    corpus=corpus,
                    document_id=document_id,
                    limit=candidate_limit,
                )
            )

        vector_lists: list[list[VectorHit]] = [[] for _ in normalized_queries]
        vector_error = ""
        embedding_batch_calls = 0
        if self.index.ready:
            try:
                vectors, embedding_batch_calls = self._query_vectors(
                    normalized_queries
                )
                vector_lists = [
                    self.index.search(
                        vector,
                        corpus=corpus,
                        document_id=document_id,
                        model=getattr(self.settings, "bailian_embedding_model", None),
                        dimension=getattr(
                            self.settings, "bailian_embedding_dimension", None
                        ),
                        source_fingerprint=self.base.source_fingerprint,
                        limit=candidate_limit,
                    )
                    for vector in vectors
                ]
            except Exception as exc:  # noqa: BLE001 - keep local FTS fallback
                vector_error = str(exc)
                logger.warning("local multi-query vector retrieval unavailable: %s", exc)

        per_query_ids: list[list[str]] = []
        for lexical, vector_hits in zip(lexical_lists, vector_lists):
            per_query_ids.append(
                self._merge_candidate_ids(
                    lexical,
                    vector_hits,
                    limit=candidate_limit,
                )
            )
        candidate_ids = list(
            dict.fromkeys(chunk_id for ids in per_query_ids for chunk_id in ids)
        )
        candidates = self.base.get_chunks_by_ids(candidate_ids)
        if not candidates:
            self.last_diagnostics = {
                "mode": "multi-query-none",
                "query_count": len(normalized_queries),
                "vector_used": any(vector_lists),
                "rerank_used": False,
                "embedding_batch_calls": embedding_batch_calls,
                "per_query_result_ids": per_query_ids,
                "query_coverage": 0.0,
                "vector_error": vector_error,
            }
            return []

        ranked = self._fuse_multi_query(
            normalized_queries,
            lexical_lists,
            vector_lists,
            candidates,
            limit=bounded_limit,
        )
        query_coverage = sum(bool(ids) for ids in per_query_ids) / len(
            per_query_ids
        )
        retriever_mode = str(
            getattr(self.settings, "rag_retriever_mode", "vector")
        ).lower()
        rerank_used = False
        adaptive_triggered = False
        rerank_margin = 0.0
        if retriever_mode in {"adaptive", "adaptive_rerank", "hybrid"}:
            margins = [
                self._vector_margin(hits)
                for hits in vector_lists
                if hits
            ]
            ambiguous = bool(margins) and min(margins) < self.adaptive_margin
            adaptive_triggered = ambiguous
            if retriever_mode == "hybrid" or ambiguous or query_coverage < 1.0:
                combined_query = "；".join(normalized_queries)
                reranked = self._try_rerank(combined_query, candidates, bounded_limit)
                rerank_margin = self._rerank_margin(reranked)
                if reranked is not None and rerank_margin >= self.adaptive_rerank_min_margin:
                    ranked_ids = {chunk.id for chunk in ranked}
                    reranked = [chunk for chunk in reranked if chunk.id in ranked_ids]
                    ranked_by_id = {chunk.id: chunk for chunk in ranked}
                    ranked = [ranked_by_id[chunk.id] for chunk in reranked]
                    ranked.extend(
                        chunk for chunk in ranked_by_id.values()
                        if chunk.id not in {item.id for item in ranked}
                    )
                    ranked = ranked[:bounded_limit]
                    rerank_used = True

        self.last_diagnostics = {
            "mode": (
                "multi-query-adaptive-rerank"
                if rerank_used
                else "multi-query-vector"
                if any(vector_lists)
                else "multi-query-fts"
            ),
            "query_count": len(normalized_queries),
            "vector_used": any(vector_lists),
            "rerank_used": rerank_used,
            "adaptive_rerank_triggered": adaptive_triggered,
            "embedding_batch_calls": embedding_batch_calls,
            "per_query_result_ids": per_query_ids,
            "query_coverage": round(query_coverage, 4),
            "rerank_margin": rerank_margin,
            "vector_error": vector_error,
        }
        return ranked[:bounded_limit]

    def _query_vectors(self, queries: Sequence[str]) -> tuple[list[list[float]], int]:
        """Load cached query vectors and batch-embed only missing values."""

        model = getattr(self.settings, "bailian_embedding_model", "")
        dimension = int(getattr(self.settings, "bailian_embedding_dimension", 0))
        vectors: list[list[float] | None] = [None] * len(queries)
        missing_indices: list[int] = []
        missing_queries: list[str] = []
        for index, query in enumerate(queries):
            cache_key = ModelResponseCache.fingerprint(
                {
                    "model": model,
                    "dimension": dimension,
                    "source_fingerprint": self.base.source_fingerprint,
                    "query": query,
                }
            )
            cached = self.cache.get("rag-query-embedding", cache_key)
            vector = _read_vector(
                cached.get("embedding") if cached else None,
                dimension,
            )
            if vector:
                vectors[index] = vector
            else:
                missing_indices.append(index)
                missing_queries.append(query)

        batch_calls = 0
        if missing_queries:
            embed_many = getattr(self.embedding_gateway, "embed_many", None)
            if callable(embed_many):
                generated = embed_many(missing_queries, text_type="query")
                batch_size = max(
                    1,
                    min(
                        int(getattr(self.settings, "bailian_embedding_batch_size", 20)),
                        20,
                    ),
                )
                batch_calls = (len(missing_queries) + batch_size - 1) // batch_size
            else:
                generated = self.embedding_gateway.embed(
                    missing_queries,
                    text_type="query",
                )
                batch_calls = 1
            if len(generated) != len(missing_indices):
                raise ValueError("Embedding 批量查询向量数量不正确。")
            for index, raw_vector in zip(missing_indices, generated):
                vector = _read_vector(raw_vector, dimension)
                if not vector:
                    raise ValueError("Embedding 批量查询向量为空或维度不正确。")
                vectors[index] = vector
                cache_key = ModelResponseCache.fingerprint(
                    {
                        "model": model,
                        "dimension": dimension,
                        "source_fingerprint": self.base.source_fingerprint,
                        "query": queries[index],
                    }
                )
                self.cache.set("rag-query-embedding", cache_key, {"embedding": vector})

        if any(vector is None for vector in vectors):
            raise ValueError("Embedding 批量查询缺少向量。")
        return [vector for vector in vectors if vector is not None], batch_calls

    @classmethod
    def _fuse_multi_query(
        cls,
        queries: Sequence[str],
        lexical_lists: Sequence[Sequence[KnowledgeChunk]],
        vector_lists: Sequence[Sequence[VectorHit]],
        candidates: Sequence[KnowledgeChunk],
        *,
        limit: int,
    ) -> list[KnowledgeChunk]:
        """Fuse per-intent rankings with weighted RRF and local MMR."""

        by_id = {chunk.id: chunk for chunk in candidates}
        fused: dict[str, float] = {chunk.id: 0.0 for chunk in candidates}
        has_vectors = any(vector_lists)
        lexical_weight = 0.05 if has_vectors else 1.0
        vector_weight = 1.0
        for query_index, (query, lexical, vector_hits) in enumerate(
            zip(queries, lexical_lists, vector_lists)
        ):
            # The first query is the broad role anchor.  It contributes recall
            # but must not drown out the card-specific intents that follow it.
            intent_weight = 0.05 if query_index == 0 else 1.0
            lexical_intent_weight = (
                0.02 if has_vectors and query_index == 0 else lexical_weight
            )
            for rank, chunk in enumerate(vector_hits, start=1):
                if chunk.chunk_id in fused:
                    fused[chunk.chunk_id] += intent_weight * vector_weight / (60.0 + rank)
            for rank, chunk in enumerate(lexical, start=1):
                if chunk.id in fused:
                    fused[chunk.id] += lexical_intent_weight / (60.0 + rank)

        ordered = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], chunk_id))
        ranked = [replace(by_id[chunk_id], score=fused[chunk_id]) for chunk_id in ordered]
        # Keep one semantic winner for every focused intent.  The first query
        # is the broad role anchor; the following queries represent concrete
        # ability cards and must retain at least one citation each whenever
        # the result budget allows it.
        pinned_ids = [
            vector_hits[0].chunk_id
            for vector_hits in vector_lists[1:]
            if vector_hits
        ]
        return cls._mmr_select(ranked, limit=limit, pinned_ids=pinned_ids)

    @classmethod
    def _mmr_select(
        cls,
        candidates: Sequence[KnowledgeChunk],
        *,
        limit: int,
        relevance_weight: float = 0.82,
        pinned_ids: Sequence[str] = (),
    ) -> list[KnowledgeChunk]:
        if len(candidates) <= limit:
            return list(candidates)
        scores = [float(chunk.score) for chunk in candidates]
        minimum = min(scores, default=0.0)
        maximum = max(scores, default=1.0)
        denominator = max(1e-9, maximum - minimum)
        pinned = set(pinned_ids)
        selected = [chunk for chunk in candidates if chunk.id in pinned][:limit]
        selected_ids = {chunk.id for chunk in selected}
        remaining = [chunk for chunk in candidates if chunk.id not in selected_ids]
        while remaining and len(selected) < limit:
            def mmr_score(chunk: KnowledgeChunk) -> tuple[float, float, str]:
                relevance = (chunk.score - minimum) / denominator
                redundancy = max(
                    (_chunk_similarity(chunk, previous) for previous in selected),
                    default=0.0,
                )
                value = relevance_weight * relevance - (1 - relevance_weight) * redundancy
                return value, chunk.score, chunk.id

            best = max(remaining, key=mmr_score)
            selected.append(best)
            remaining.remove(best)
        return selected

    @staticmethod
    def _merge_candidate_ids(
        lexical: Sequence[KnowledgeChunk],
        vector_hits: Sequence[VectorHit],
        *,
        limit: int,
    ) -> list[str]:
        """Interleave lexical and vector recall instead of truncating lexical first.

        The old lexical-first truncation could remove semantically relevant
        vector candidates before reranking.  Alternating both ranked lists
        preserves recall while keeping the remote rerank input bounded.
        """

        lexical_ids = [chunk.id for chunk in lexical]
        vector_ids = [hit.chunk_id for hit in vector_hits]
        merged: list[str] = []
        seen: set[str] = set()
        width = max(len(lexical_ids), len(vector_ids))
        for index in range(width):
            for values in (vector_ids, lexical_ids):
                if index >= len(values):
                    continue
                chunk_id = values[index]
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                merged.append(chunk_id)
                if len(merged) >= limit:
                    return merged
        return merged

    @staticmethod
    def _vector_ranked(
        vector_hits: Sequence[VectorHit],
        candidates: Sequence[KnowledgeChunk],
        lexical: Sequence[KnowledgeChunk],
        *,
        limit: int,
    ) -> list[KnowledgeChunk]:
        by_id = {chunk.id: chunk for chunk in candidates}
        ranked = [
            replace(by_id[hit.chunk_id], score=hit.score)
            for hit in vector_hits
            if hit.chunk_id in by_id
        ]
        seen_ids = {chunk.id for chunk in ranked}
        if len(ranked) < limit:
            ranked.extend(
                replace(chunk, score=chunk.score)
                for chunk in lexical
                if chunk.id not in seen_ids
            )
        return ranked[:limit]

    @staticmethod
    def _vector_margin(vector_hits: Sequence[VectorHit]) -> float:
        if len(vector_hits) < 2:
            return float("inf")
        return max(0.0, float(vector_hits[0].score) - float(vector_hits[1].score))

    @staticmethod
    def _rerank_margin(reranked: Sequence[KnowledgeChunk] | None) -> float:
        if not reranked or len(reranked) < 2:
            return 0.0
        return max(0.0, float(reranked[0].score) - float(reranked[1].score))

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


_ASCII_TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]*")
_CJK_TOKEN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]{2,}")


def _tokenize(value: str) -> set[str]:
    """Create a small deterministic token set for heading/MMR comparisons."""

    tokens = {item.lower() for item in _ASCII_TOKEN_RE.findall(value)}
    for run in _CJK_TOKEN_RE.findall(value):
        tokens.add(run)
        tokens.update(run[index : index + 2] for index in range(len(run) - 1))
    return {token for token in tokens if len(token) >= 2}


def _chunk_similarity(left: KnowledgeChunk, right: KnowledgeChunk) -> float:
    left_tokens = _tokenize(" ".join(left.heading_path) + " " + left.content)
    right_tokens = _tokenize(" ".join(right.heading_path) + " " + right.content)
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / max(1, len(union))


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
