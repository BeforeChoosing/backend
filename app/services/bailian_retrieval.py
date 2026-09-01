"""Bailian embedding and reranking gateways used by local RAG.

The knowledge corpus and its vector index stay on the local machine.  Only
the text needed to create an embedding or to rerank a small candidate set is
sent to the configured Bailian endpoint.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Sequence

from app.config import Settings
from app.services.llm_gateway import LLMGatewayCancelledError, LLMGatewayError
from app.services.audit_log import record_model_call
from app.services.llm_request_queue import LLMRequestCancelled, get_llm_request_queue
from app.services.request_context import get_request_context


class BailianRetrievalError(LLMGatewayError):
    """Base error for Bailian retrieval APIs."""


class EmbeddingGatewayError(BailianRetrievalError):
    """Raised when the embedding API cannot return valid vectors."""


class RerankGatewayError(BailianRetrievalError):
    """Raised when the rerank API cannot return valid scores."""


@dataclass(frozen=True)
class RerankHit:
    index: int
    relevance_score: float


class DashScopeEmbeddingGateway:
    """Call a Bailian text embedding endpoint without downloading a model."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def embed(
        self,
        texts: Sequence[str],
        *,
        text_type: str = "document",
    ) -> list[list[float]]:
        normalized = [str(text).strip() for text in texts]
        if not normalized or any(not text for text in normalized):
            raise EmbeddingGatewayError("Embedding 输入不能为空。")
        if len(normalized) > self.batch_size:
            raise EmbeddingGatewayError(
                f"Embedding 单次最多接收 {self.batch_size} 条文本。"
            )
        if text_type not in {"query", "document"}:
            raise EmbeddingGatewayError("Embedding text_type 只能是 query 或 document。")
        self._require_key()

        payload = self._payload(normalized, text_type=text_type)
        started = time.perf_counter()
        response_payload = _post_json(
            self.settings.bailian_embedding_url, self.settings.dashscope_api_key, payload,
            self.settings.request_timeout_seconds, service_name="Embedding",
            error_type=EmbeddingGatewayError,
            max_concurrency=getattr(self.settings, "llm_max_concurrency", 2),
            max_requests_per_minute=getattr(
                self.settings, "llm_max_requests_per_minute", 30
            ),
            model_max_concurrency=getattr(
                self.settings, "llm_model_max_concurrency", 1
            ),
        )
        record_model_call(
            getattr(self.settings, "profile_db_path", "profile.db"), service="embedding",
            model=self.settings.bailian_embedding_model,
            duration_ms=(time.perf_counter() - started) * 1000,
            metadata={"endpoint": "embedding", "items": len(normalized)},
        )
        return self._parse_vectors(response_payload, len(normalized))

    def embed_many(
        self,
        texts: Sequence[str],
        *,
        text_type: str = "document",
    ) -> list[list[float]]:
        """Embed an arbitrary number of texts in API-sized batches."""
        values = list(texts)
        if not values:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(values), self.batch_size):
            vectors.extend(
                self.embed(
                    values[start : start + self.batch_size],
                    text_type=text_type,
                )
            )
        return vectors

    @property
    def batch_size(self) -> int:
        configured = int(self.settings.bailian_embedding_batch_size)
        return max(1, min(configured, 20))

    def _payload(self, texts: list[str], *, text_type: str) -> dict[str, Any]:
        # The direct DashScope service uses input.texts and parameters.dimension.
        # Keep compatibility with an OpenAI-compatible /embeddings URL when a
        # team overrides BAILIAN_EMBEDDING_URL for an alternate deployment.
        if "/compatible-mode/" in self.settings.bailian_embedding_url:
            return {
                "model": self.settings.bailian_embedding_model,
                "input": texts,
                "dimensions": self.settings.bailian_embedding_dimension,
                "encoding_format": "float",
            }
        return {
            "model": self.settings.bailian_embedding_model,
            "input": {"texts": texts},
            "parameters": {
                "dimension": self.settings.bailian_embedding_dimension,
                "output_type": "dense",
                "text_type": text_type,
            },
        }

    def _require_key(self) -> None:
        if not self.settings.dashscope_api_key:
            raise EmbeddingGatewayError(
                "未配置 DASHSCOPE_API_KEY，无法调用百炼 Embedding。"
            )

    def _parse_vectors(
        self,
        payload: dict[str, Any],
        expected_count: int,
    ) -> list[list[float]]:
        output = payload.get("output")
        items: Any = output.get("embeddings") if isinstance(output, dict) else None
        if not isinstance(items, list):
            items = payload.get("data")
        if not isinstance(items, list) or len(items) != expected_count:
            raise EmbeddingGatewayError("百炼 Embedding 响应缺少完整的向量列表。")

        vectors: list[list[float] | None] = [None] * expected_count
        for fallback_index, item in enumerate(items):
            if not isinstance(item, dict):
                raise EmbeddingGatewayError("百炼 Embedding 响应包含无效向量项。")
            raw_index = item.get("text_index", item.get("index", fallback_index))
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise EmbeddingGatewayError("百炼 Embedding 响应包含无效文本序号。")
            if raw_index < 0 or raw_index >= expected_count or vectors[raw_index] is not None:
                raise EmbeddingGatewayError("百炼 Embedding 响应的文本序号不完整。")
            raw_vector = item.get("embedding")
            if not isinstance(raw_vector, list):
                raise EmbeddingGatewayError("百炼 Embedding 响应包含无效向量。")
            vector: list[float] = []
            for value in raw_vector:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise EmbeddingGatewayError("百炼 Embedding 响应包含非数字向量值。")
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise EmbeddingGatewayError("百炼 Embedding 响应包含非有限向量值。")
                vector.append(numeric)
            expected_dimension = int(self.settings.bailian_embedding_dimension)
            if expected_dimension > 0 and len(vector) != expected_dimension:
                raise EmbeddingGatewayError(
                    "百炼 Embedding 向量维度与 BAILIAN_EMBEDDING_DIMENSION 不一致。"
                )
            vectors[raw_index] = vector

        if any(vector is None for vector in vectors):
            raise EmbeddingGatewayError("百炼 Embedding 响应缺少文本序号。")
        return [vector for vector in vectors if vector is not None]


class DashScopeRerankGateway:
    """Call Bailian text rerank for a query and a bounded candidate set."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int = 5,
    ) -> list[RerankHit]:
        normalized_query = str(query).strip()
        normalized_documents = [str(document).strip() for document in documents]
        if not normalized_query:
            raise RerankGatewayError("Rerank query 不能为空。")
        if not normalized_documents or any(not document for document in normalized_documents):
            raise RerankGatewayError("Rerank documents 不能为空。")
        self._require_key()
        bounded_top_n = max(1, min(int(top_n), len(normalized_documents)))
        payload = {
            "model": self.settings.bailian_rerank_model,
            "input": {
                "query": normalized_query,
                "documents": normalized_documents,
            },
            "parameters": {
                "return_documents": True,
                "top_n": bounded_top_n,
            },
        }
        started = time.perf_counter()
        response_payload = _post_json(
            self.settings.bailian_rerank_url,
            self.settings.dashscope_api_key,
            payload,
            self.settings.request_timeout_seconds,
            service_name="Rerank",
            error_type=RerankGatewayError,
            max_concurrency=getattr(self.settings, "llm_max_concurrency", 2),
            max_requests_per_minute=getattr(
                self.settings, "llm_max_requests_per_minute", 30
            ),
            model_max_concurrency=getattr(
                self.settings, "llm_model_max_concurrency", 1
            ),
        )
        record_model_call(
            getattr(self.settings, "profile_db_path", "profile.db"), service="rerank",
            model=self.settings.bailian_rerank_model,
            duration_ms=(time.perf_counter() - started) * 1000,
            metadata={"endpoint": "rerank", "documents": len(normalized_documents)},
        )
        output = response_payload.get("output")
        raw_results: Any = output.get("results") if isinstance(output, dict) else None
        if not isinstance(raw_results, list):
            raw_results = response_payload.get("results")
        if not isinstance(raw_results, list):
            raise RerankGatewayError("百炼 Rerank 响应缺少结果列表。")

        hits: list[RerankHit] = []
        for item in raw_results:
            if not isinstance(item, dict):
                raise RerankGatewayError("百炼 Rerank 响应包含无效结果项。")
            index = item.get("index")
            score = item.get("relevance_score", item.get("score"))
            if isinstance(index, bool) or not isinstance(index, int):
                raise RerankGatewayError("百炼 Rerank 响应包含无效文档序号。")
            if index < 0 or index >= len(normalized_documents):
                raise RerankGatewayError("百炼 Rerank 响应的文档序号越界。")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise RerankGatewayError("百炼 Rerank 响应包含无效相关性分数。")
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise RerankGatewayError("百炼 Rerank 响应包含非有限相关性分数。")
            hits.append(RerankHit(index=index, relevance_score=numeric_score))

        hits.sort(key=lambda hit: (-hit.relevance_score, hit.index))
        return hits[:bounded_top_n]

    def _require_key(self) -> None:
        if not self.settings.dashscope_api_key:
            raise RerankGatewayError("未配置 DASHSCOPE_API_KEY，无法调用百炼 Rerank。")


def _post_json(
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
    *,
    service_name: str,
    error_type: type[BailianRetrievalError],
    max_concurrency: int,
    max_requests_per_minute: int,
    model_max_concurrency: int = 1,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    context = get_request_context()
    queue = get_llm_request_queue(
        max_concurrency=max_concurrency,
        max_requests_per_minute=max_requests_per_minute,
        model_max_concurrency=model_max_concurrency,
    )
    try:
        with queue.admission(request_id=context.request_id, user_id=context.user_id):
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw_body = response.read().decode("utf-8")
    except LLMRequestCancelled as exc:
        raise LLMGatewayCancelledError(str(exc)) from exc
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise error_type(f"百炼 {service_name} 请求失败（HTTP {exc.code}）：{body}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise error_type(f"百炼 {service_name} 请求超时或无法连接：{exc}") from exc

    try:
        parsed = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise error_type(f"百炼 {service_name} 返回的响应不是合法 JSON。") from exc
    if not isinstance(parsed, dict):
        raise error_type(f"百炼 {service_name} 返回的 JSON 顶层必须是对象。")
    return parsed
