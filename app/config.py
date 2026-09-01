from dataclasses import dataclass
import os
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv

load_dotenv()


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


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
_DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS = 180


@dataclass(frozen=True)
class Settings:
    app_name: str = "选择之前 API"
    api_prefix: str = "/api/v1"
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    dashscope_base_url: str = _CHAT_URL
    qwen_model: str = os.getenv("QWEN_MODEL", "qwen-plus")
    trial_base_model: str = os.getenv("TRIAL_BASE_MODEL", "")
    trial_sft_model: str = os.getenv("TRIAL_SFT_MODEL", "")
    trial_verifier_model: str = os.getenv("TRIAL_VERIFIER_MODEL", "")
    trial_teacher_model: str = os.getenv("TRIAL_TEACHER_MODEL", "qwen3-vl-plus")
    trial_review_model: str = os.getenv(
        "TRIAL_REVIEW_MODEL", "qwen3-vl-235b-a22b-instruct"
    )
    trial_teacher_prompt_version: str = os.getenv(
        "TRIAL_TEACHER_PROMPT_VERSION", "trial-teacher-v1"
    )
    trial_teacher_cache_path: str = os.getenv(
        "TRIAL_TEACHER_CACHE_PATH", "datasets/trial_agent/v1/teacher_cache.sqlite3"
    )
    trial_verifier_min_evidence_coverage: float = float(
        os.getenv("TRIAL_VERIFIER_MIN_EVIDENCE_COVERAGE", "0.75")
    )
    bailian_vision_model: str = os.getenv("BAILIAN_VISION_MODEL", "qwen-vl-ocr")
    multimodal_max_pages: int = int(os.getenv("MULTIMODAL_MAX_PAGES", "8"))
    # qwen3.6-plus can legitimately take around 90s for structured profile
    # proposals; keep enough headroom for the upstream response before
    # returning a 504 to the client.
    request_timeout_seconds: float = float(
        os.getenv("LLM_REQUEST_TIMEOUT", str(_DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS))
    )
    llm_max_concurrency: int = int(os.getenv("LLM_MAX_CONCURRENCY", "2"))
    llm_max_requests_per_minute: int = int(
        os.getenv("LLM_MAX_REQUESTS_PER_MINUTE", "30")
    )
    profile_db_path: str = os.getenv("PROFILE_DB_PATH", "profile.db")
    auth_session_ttl_hours: int = int(os.getenv("AUTH_SESSION_TTL_HOURS", str(24 * 30)))
    knowledge_dir: str = os.getenv("KNOWLEDGE_DIR", "knowledge/public")
    knowledge_db_path: str = os.getenv("KNOWLEDGE_DB_PATH", "knowledge.db")
    bailian_embedding_url: str = _env_or_default(
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
    bailian_rerank_url: str = _env_or_default(
        "BAILIAN_RERANK_URL",
        _service_url(_CHAT_URL, "/api/v1/services/rerank/text-rerank/text-rerank"),
    )
    bailian_rerank_model: str = os.getenv(
        "BAILIAN_RERANK_MODEL", "qwen3-rerank"
    )
    bailian_rerank_fallback_model: str = os.getenv(
        "BAILIAN_RERANK_FALLBACK_MODEL", "gte-rerank-v2"
    )
    # Vector is the current best-performing default on the expanded set.
    # Adaptive is an opt-in cost-aware rerank experiment; hybrid reproduces
    # the fixed fusion + rerank ablation arm.
    rag_retriever_mode: str = os.getenv("RAG_RETRIEVER_MODE", "vector")
    rag_candidate_limit: int = int(os.getenv("RAG_CANDIDATE_LIMIT", "20"))
    rag_rerank_limit: int = int(os.getenv("RAG_RERANK_LIMIT", "5"))
    # Tuned on scripts/rag_eval_cases_v3.json: a shorter RRF tail and a
    # relevance-heavy MMR pass improve MRR without changing Hit@1/Hit@5.
    rag_rrf_k: float = float(os.getenv("RAG_RRF_K", "20"))
    rag_rrf_anchor_lexical_weight: float = float(
        os.getenv("RAG_RRF_ANCHOR_LEXICAL_WEIGHT", "0")
    )
    rag_mmr_relevance_weight: float = float(
        os.getenv("RAG_MMR_RELEVANCE_WEIGHT", "0.65")
    )
    rag_adaptive_margin: float = float(os.getenv("RAG_ADAPTIVE_MARGIN", "0.055"))
    rag_adaptive_rerank_min_margin: float = float(
        os.getenv("RAG_ADAPTIVE_RERANK_MIN_MARGIN", "0.02")
    )
    cors_origins: tuple[str, ...] = _csv(
        os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    )

    @property
    def qwen_configured(self) -> bool:
        return bool(self.dashscope_api_key)


def get_settings() -> Settings:
    return Settings()
