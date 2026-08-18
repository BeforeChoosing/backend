from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    app_name: str = "选择之前 API"
    api_prefix: str = "/api/v1"
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    dashscope_base_url: str = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    )
    qwen_model: str = os.getenv("QWEN_MODEL", "qwen-plus")
    request_timeout_seconds: float = float(os.getenv("LLM_REQUEST_TIMEOUT", "45"))
    cors_origins: tuple[str, ...] = _csv(
        os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    )

    @property
    def qwen_configured(self) -> bool:
        return bool(self.dashscope_api_key)


def get_settings() -> Settings:
    return Settings()
