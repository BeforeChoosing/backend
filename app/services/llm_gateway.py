import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

from app.config import Settings
from app.services.audit_log import record_model_call


class LLMGatewayError(RuntimeError):
    """Raised when the configured Qwen gateway cannot produce a response."""


class DashScopeQwenGateway:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str | None = None,
    ) -> dict[str, Any]:
        if not self.settings.qwen_configured:
            raise LLMGatewayError(
                "未配置 DASHSCOPE_API_KEY。请在 backend/.env 中配置百炼密钥，"
                "不会使用伪造结果代替模型响应。"
            )

        payload = {
            "model": model or self.settings.qwen_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            self.settings.dashscope_base_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.dashscope_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(
                request, timeout=self.settings.request_timeout_seconds
            ) as response:
                raw_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            raise LLMGatewayError(f"百炼请求失败（HTTP {exc.code}）：{body}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMGatewayError(f"百炼请求超时或无法连接：{exc}") from exc

        try:
            response_payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise LLMGatewayError("百炼返回的响应不是合法 JSON。") from exc

        usage = response_payload.get("usage") if isinstance(response_payload, dict) else None
        record_model_call(
            getattr(self.settings, "profile_db_path", "profile.db"),
            service="qwen",
            model=model or self.settings.qwen_model,
            duration_ms=(time.perf_counter() - started) * 1000,
            input_tokens=_usage_int(usage, "prompt_tokens", "input_tokens"),
            output_tokens=_usage_int(usage, "completion_tokens", "output_tokens"),
            metadata={"endpoint": "chat"},
        )
        content = self._extract_content(response_payload)
        return self._parse_json_content(content)
    @staticmethod
    def _extract_content(payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, list):
                content = "".join(
                    item.get("text", "") for item in content if isinstance(item, dict)
                )
            if isinstance(content, str) and content.strip():
                return content

        output = payload.get("output") or {}
        text = output.get("text")
        if isinstance(text, str) and text.strip():
            return text

        raise LLMGatewayError("百炼响应中没有可读取的模型文本。")

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, Any]:
        cleaned = content.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMGatewayError("Qwen 输出无法解析为 JSON。") from exc
        if not isinstance(parsed, dict):
            raise LLMGatewayError("Qwen 输出的 JSON 顶层必须是对象。")
        return parsed


def _usage_int(usage: Any, *keys: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None
