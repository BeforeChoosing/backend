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


def test_local_retriever_fuses_separate_intents_and_exposes_coverage(tmp_path):
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
    results = retriever.search_many(
        ["AI 产品经理 用户研究", "AI 产品经理 技术落地"],
        corpus="career",
        document_id="job-ai-product-manager-v1",
        limit=2,
    )

    assert {chunk.heading_path[-1] for chunk in results} == {"用户研究", "技术落地"}
    assert retriever.last_diagnostics["mode"] == "multi-query-fts"
    assert retriever.last_diagnostics["query_coverage"] == 1.0


def test_local_retriever_keeps_manifest_provenance_fields(tmp_path):
    source_dir = tmp_path / "knowledge"
    source_dir.mkdir()
    (source_dir / "source.md").write_text(
        "# 有来源资料\n\n## 章节一\n\n官方资料摘要。\n",
        encoding="utf-8",
    )
    (source_dir / "manifest.json").write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "path": "source.md",
                        "document_id": "source-v1",
                        "corpus": "method",
                        "trust_level": "primary_official",
                        "source_note": "官方原文核验",
                        "source_url": "https://example.com/source",
                        "published_at": "2026-08-01",
                        "retrieved_at": "2026-08-29",
                        "version": "v1.2",
                        "license": "公开网页，按引用规范使用",
                        "source_type": "official_documentation",
                        "authority_score": 1.0,
                        "relevance_score": 0.9,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    chunk = KnowledgeRetriever(source_dir, tmp_path / "knowledge.db").list_chunks()[0]

    assert chunk.source_url == "https://example.com/source"
    assert chunk.version == "v1.2"
    assert chunk.authority_score == 1.0
    assert chunk.relevance_score == 0.9
