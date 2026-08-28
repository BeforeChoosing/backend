from dataclasses import dataclass
import os
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv

load_dotenv()


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _service_url(base_url: str, path: str) -> str:
    """Derive a DashScope service endpoint from the configured chat host."""
    parsed = urlsplit(base_url)
    if not parsed.scheme or not parsed.netloc:
        return path
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


_DEFAULT_CHAT_URL = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
)
_CHAT_URL = os.getenv("DASHSCOPE_BASE_URL", _DEFAULT_CHAT_URL)


@dataclass(frozen=True)
class Settings:
    app_name: str = "选择之前 API"
    api_prefix: str = "/api/v1"
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    dashscope_base_url: str = _CHAT_URL
    qwen_model: str = os.getenv("QWEN_MODEL", "qwen-plus")
    request_timeout_seconds: float = float(os.getenv("LLM_REQUEST_TIMEOUT", "45"))
    profile_db_path: str = os.getenv("PROFILE_DB_PATH", "profile.db")
    knowledge_dir: str = os.getenv("KNOWLEDGE_DIR", "knowledge/public")
    knowledge_db_path: str = os.getenv("KNOWLEDGE_DB_PATH", "knowledge.db")
    bailian_embedding_url: str = os.getenv(
        "BAILIAN_EMBEDDING_URL",
        _service_url(
            _CHAT_URL,
            "/api/v1/services/embeddings/text-embedding/text-embedding",
        ),
    )
    bailian_embedding_model: str = os.getenv(
        "BAILIAN_EMBEDDING_MODEL", "qwen3.7-text-embedding"
    )
    bailian_embedding_dimension: int = int(
        os.getenv("BAILIAN_EMBEDDING_DIMENSION", "1024")
    )
    bailian_embedding_batch_size: int = int(
        os.getenv("BAILIAN_EMBEDDING_BATCH_SIZE", "20")
    )
    bailian_rerank_url: str = os.getenv(
        "BAILIAN_RERANK_URL",
        _service_url(_CHAT_URL, "/api/v1/services/rerank/text-rerank/text-rerank"),
    )
    bailian_rerank_model: str = os.getenv(
        "BAILIAN_RERANK_MODEL", "qwen3-rerank"
    )
    bailian_rerank_fallback_model: str = os.getenv(
        "BAILIAN_RERANK_FALLBACK_MODEL", "gte-rerank-v2"
    )
    rag_retriever_mode: str = os.getenv("RAG_RETRIEVER_MODE", "hybrid")
    rag_candidate_limit: int = int(os.getenv("RAG_CANDIDATE_LIMIT", "20"))
    rag_rerank_limit: int = int(os.getenv("RAG_RERANK_LIMIT", "5"))
    cors_origins: tuple[str, ...] = _csv(
        os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    )

    @property
    def qwen_configured(self) -> bool:
        return bool(self.dashscope_api_key)


def get_settings() -> Settings:
    return Settings()
