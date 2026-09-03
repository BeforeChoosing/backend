import json

from app.knowledge.retriever import KnowledgeRetriever
from app.knowledge.vector_index import LocalVectorIndex, VectorIndexBuilder


class _FakeEmbeddingGateway:
    def __init__(self):
        self.calls = []

    def embed_many(self, texts, *, text_type):
        self.calls.append((list(texts), text_type))
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "用户" in lowered else 0.0,
                    1.0 if "模型" in lowered else 0.0,
                    1.0 if "落地" in lowered else 0.0,
                ]
            )
        return vectors


def _retriever(tmp_path):
    source_dir = tmp_path / "knowledge"
    (source_dir / "jobs").mkdir(parents=True)
    (source_dir / "jobs" / "ai_product_manager.md").write_text(
        "# AI 产品经理\n\n## 用户研究\n\n通过访谈识别用户问题。\n\n"
        "## 技术落地\n\n负责模型评测和方案落地。\n",
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


def test_vector_index_builds_once_and_reuses_unchanged_chunks(tmp_path):
    retriever = _retriever(tmp_path)
    gateway = _FakeEmbeddingGateway()
    builder = VectorIndexBuilder(
        retriever,
        gateway,
        model="qwen3.7-text-embedding",
        dimension=3,
        batch_size=20,
    )

    first = builder.build()
    second = builder.build()

    assert first.embedded == 2
    assert first.reused == 0
    assert second.embedded == 0
    assert second.reused == 2
    assert len(gateway.calls) == 1
    assert gateway.calls[0][1] == "document"

    index = LocalVectorIndex(retriever.db_path)
    hits = index.search(
        [1.0, 0.0, 0.0],
        corpus="career",
        document_id="job-ai-product-manager-v1",
        limit=2,
    )
    assert hits
    assert hits[0].chunk_id.startswith("chk-")
    assert hits[0].score > 0.9


def test_rebuild_preserves_vectors_for_unchanged_chunks(tmp_path):
    retriever = _retriever(tmp_path)
    gateway = _FakeEmbeddingGateway()
    builder = VectorIndexBuilder(
        retriever,
        gateway,
        model="qwen3.7-text-embedding",
        dimension=3,
        batch_size=20,
    )

    first = builder.build()
    assert first.embedded == 2

    # Adding a document changes the corpus fingerprint. The two existing
    # chunks must survive the Markdown rebuild and remain reusable.
    (retriever.source_dir / "jobs" / "new_role.md").write_text(
        "# 新岗位\n\n## 平台\n\n负责模型落地。\n",
        encoding="utf-8",
    )
    refreshed = KnowledgeRetriever(retriever.source_dir, retriever.db_path)
    second = VectorIndexBuilder(
        refreshed,
        gateway,
        model="qwen3.7-text-embedding",
        dimension=3,
        batch_size=20,
    ).build()

    assert second.total == 3
    assert second.reused == 2
    assert second.embedded == 1
    assert len(gateway.calls) == 2
    assert LocalVectorIndex(refreshed.db_path).count == 3
