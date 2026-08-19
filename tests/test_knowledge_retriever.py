import json

from app.knowledge.retriever import KnowledgeRetriever


def test_local_retriever_indexes_markdown_and_returns_citations(tmp_path):
    source_dir = tmp_path / "knowledge"
    (source_dir / "jobs").mkdir(parents=True)
    (source_dir / "jobs" / "ai_product_manager.md").write_text(
        "# AI 产品经理\n\n## 用户研究\n\n通过访谈和反馈识别用户问题。\n\n"
        "## 技术落地\n\n负责模型评测、方案验证和跨团队落地。\n",
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

    retriever = KnowledgeRetriever(source_dir, tmp_path / "knowledge.db")

    assert retriever.chunk_count == 2
    results = retriever.search(
        "用户研究和模型评测",
        corpus="career",
        document_id="job-ai-product-manager-v1",
        limit=5,
    )

    assert results
    assert results[0].document_id == "job-ai-product-manager-v1"
    assert results[0].source_locator.startswith("jobs/ai_product_manager.md#")
    assert results[0].trust_level == "secondary_summary"
    assert results[0].source_note == "测试资料"
