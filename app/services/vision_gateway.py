from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Sequence

from app.config import Settings
from app.services.llm_gateway import DashScopeQwenGateway, LLMGatewayError
from app.services.audit_log import record_model_call


@dataclass(frozen=True)
class VisionImage:
    page: int
    data: bytes
    mime_type: str


class DashScopeVisionGateway:
    """Call a Bailian Qwen-VL compatible endpoint with in-memory images."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Sequence[VisionImage],
        *,
        model: str | None = None,
    ) -> dict[str, Any]:
        if not self.settings.qwen_configured:
            raise LLMGatewayError("未配置 DASHSCOPE_API_KEY，无法调用百炼视觉模型。")
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image in images:
            encoded = base64.b64encode(image.data).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image.mime_type};base64,{encoded}",
                    },
                }
            )
        payload = {
            "model": model or self.settings.bailian_vision_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": 0.1,
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
            raise LLMGatewayError(f"百炼视觉请求失败（HTTP {exc.code}）：{body}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMGatewayError(f"百炼视觉请求超时或无法连接：{exc}") from exc
        try:
            response_payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise LLMGatewayError("百炼视觉返回的响应不是合法 JSON。") from exc
        usage = response_payload.get("usage") if isinstance(response_payload, dict) else None
        record_model_call(
            getattr(self.settings, "profile_db_path", "profile.db"),
            service="qwen-vl",
            model=model or self.settings.bailian_vision_model,
            duration_ms=(time.perf_counter() - started) * 1000,
            input_tokens=_usage_int(usage, "prompt_tokens", "input_tokens"),
            output_tokens=_usage_int(usage, "completion_tokens", "output_tokens"),
            metadata={"endpoint": "vision", "images": len(images)},
        )
        content_text = DashScopeQwenGateway._extract_content(response_payload)
        return DashScopeQwenGateway._parse_json_content(content_text)


def _usage_int(usage: Any, *keys: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None
