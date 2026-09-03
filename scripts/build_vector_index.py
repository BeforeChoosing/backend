"""Build the local vector index through the configured Bailian embedding API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.config import get_settings  # noqa: E402
from app.knowledge.retriever import KnowledgeRetriever  # noqa: E402
from app.knowledge.vector_index import VectorIndexBuilder  # noqa: E402
from app.services.bailian_retrieval import DashScopeEmbeddingGateway  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用百炼 Embedding 建立本地向量索引")
    parser.add_argument("--corpus", default=None, help="只索引指定 corpus，例如 career")
    parser.add_argument("--document-id", default=None, help="只索引指定文档")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    try:
        retriever = KnowledgeRetriever(settings.knowledge_dir, settings.knowledge_db_path)
        report = VectorIndexBuilder(
            retriever,
            DashScopeEmbeddingGateway(settings),
            model=settings.bailian_embedding_model,
            dimension=settings.bailian_embedding_dimension,
            batch_size=settings.bailian_embedding_batch_size,
        ).build(corpus=args.corpus, document_id=args.document_id)
    except Exception as exc:  # noqa: BLE001 - present actionable CLI diagnostics
        print(f"向量索引构建失败：{exc}", file=sys.stderr)
        return 1

    print(
        "向量索引完成："
        f"总片段 {report.total}，新生成 {report.embedded}，复用 {report.reused}，"
        f"清理 {report.removed}；模型 {settings.bailian_embedding_model}。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
