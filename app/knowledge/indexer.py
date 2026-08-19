"""Build the local SQLite FTS5 index used by CareerAgent."""

from app.config import get_settings
from app.knowledge.retriever import KnowledgeRetriever


def main() -> None:
    settings = get_settings()
    retriever = KnowledgeRetriever(settings.knowledge_dir, settings.knowledge_db_path)
    chunk_count = retriever.rebuild()
    print(f"已建立本地岗位知识库索引：{chunk_count} 个片段")


if __name__ == "__main__":
    main()
