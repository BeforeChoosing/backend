import json
from types import SimpleNamespace

from app.knowledge.retriever import KnowledgeRetriever
from app.knowledge.vector_index import VectorIndexBuilder
from app.knowledge.hybrid import HybridKnowledgeRetriever
from app.services.model_response_cache import ModelResponseCache


class _FakeEmbeddingGateway:
    def __init__(self):
        self.calls = []

    def embed(self, texts, *, text_type):
        self.calls.append((list(texts), text_type))
        return [
            [
                1.0 if "用户" in text else 0.0,
                1.0 if "模型" in text else 0.0,
            ]
            for text in texts
        ]

    def embed_many(self, texts, *, text_type):
        return self.embed(texts, text_type=text_type)


class _FakeRerankGateway:
    def __init__(self):
        self.calls = []

    def rerank(self, query, documents, *, top_n):
        self.calls.append((query, list(documents), top_n))
        # Prefer the document whose text contains the query's second concept.
        order = sorted(
            range(len(documents)),
            key=lambda index: ("模型" not in documents[index], index),
        )
        return [
            SimpleNamespace(index=index, relevance_score=1.0 - position * 0.1)
            for position, index in enumerate(order[:top_n])
        ]


class _FailingRerankGateway:
    def rerank(self, query, documents, *, top_n):
        raise RuntimeError("模拟远端重排不可用")


def _retriever(tmp_path):
    source_dir = tmp_path / "knowledge"
    (source_dir / "jobs").mkdir(parents=True)
    (source_dir / "jobs" / "ai_product_manager.md").write_text(
        "# AI 产品经理\n\n## 用户研究\n\n用户访谈与问题洞察。\n\n"
        "## 技术落地\n\n模型评测与技术落地。\n\n"
        "## 组合方案\n\n用户研究和模型落地需要协同。\n",
        encoding="utf-8",
    )
    (source_dir / "manifest.json").write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "path": "jobs/ai_product_manager.md",
                        "document_id": "job-ai-product-manager-v1",
                        "corpus": "career",
                        "trust_level": "secondary_summary",
                        "source_note": "测试资料",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return KnowledgeRetriever(source_dir, tmp_path / "knowledge.db")


def _settings():
    return SimpleNamespace(
        dashscope_api_key="test-key",
        bailian_embedding_model="qwen3.7-text-embedding",
        bailian_embedding_dimension=2,
        bailian_rerank_model="gte-rerank-v2",
        rag_candidate_limit=20,
        rag_rerank_limit=2,
    )


def test_hybrid_retriever_uses_local_vectors_rerank_and_cache(tmp_path):
    retriever = _retriever(tmp_path)
    settings = _settings()
    embedding = _FakeEmbeddingGateway()
    VectorIndexBuilder(
        retriever,
        embedding,
        model=settings.bailian_embedding_model,
        dimension=settings.bailian_embedding_dimension,
    ).build()
    query_embedding_calls_before_search = len(embedding.calls)
    rerank = _FakeRerankGateway()
    hybrid = HybridKnowledgeRetriever(
        retriever.source_dir,
        retriever.db_path,
        settings=settings,
        embedding_gateway=embedding,
        rerank_gateway=rerank,
        cache=ModelResponseCache(retriever.db_path),
    )

    first = hybrid.search(
        "用户和模型如何落地",
        corpus="career",
        document_id="job-ai-product-manager-v1",
        limit=2,
    )
    second = hybrid.search(
        "用户和模型如何落地",
        corpus="career",
        document_id="job-ai-product-manager-v1",
        limit=2,
    )

    assert len(first) == len(second) == 2
    assert first[0].score > first[1].score
    assert "模型" in first[0].content
    assert len(embedding.calls) == query_embedding_calls_before_search + 1
    assert len(rerank.calls) == 1


def test_hybrid_retriever_keeps_deterministic_fallback_when_rerank_fails(tmp_path):
    retriever = _retriever(tmp_path)
    settings = _settings()
    embedding = _FakeEmbeddingGateway()
    VectorIndexBuilder(
        retriever,
        embedding,
        model=settings.bailian_embedding_model,
        dimension=settings.bailian_embedding_dimension,
    ).build()
    hybrid = HybridKnowledgeRetriever(
        retriever.source_dir,
        retriever.db_path,
        settings=settings,
        embedding_gateway=embedding,
        rerank_gateway=_FailingRerankGateway(),
        cache=ModelResponseCache(retriever.db_path),
    )

    results = hybrid.search("用户和模型如何落地", corpus="career", limit=2)

    assert len(results) == 2
    assert hybrid.last_diagnostics["mode"] == "hybrid"
    assert hybrid.last_diagnostics["rerank_used"] is False


def test_vector_mode_uses_local_semantic_order_without_rerank(tmp_path):
    retriever = _retriever(tmp_path)
    settings = SimpleNamespace(**vars(_settings()), rag_retriever_mode="vector")
    embedding = _FakeEmbeddingGateway()
    VectorIndexBuilder(
        retriever,
        embedding,
        model=settings.bailian_embedding_model,
        dimension=settings.bailian_embedding_dimension,
    ).build()
    rerank = _FailingRerankGateway()
    hybrid = HybridKnowledgeRetriever(
        retriever.source_dir,
        retriever.db_path,
        settings=settings,
        embedding_gateway=embedding,
        rerank_gateway=rerank,
        cache=ModelResponseCache(retriever.db_path),
    )

    results = hybrid.search("用户和模型如何落地", corpus="career", limit=2)

    assert len(results) == 2
    assert hybrid.last_diagnostics["mode"] == "vector"
    assert hybrid.last_diagnostics["rerank_used"] is False
