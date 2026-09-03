from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Sequence

from app.config import Settings
from app.services.llm_gateway import (
    DashScopeQwenGateway,
    LLMGatewayCancelledError,
    LLMGatewayError,
    LLMGatewayTimeoutError,
)
from app.services.audit_log import record_model_call
from app.services.llm_request_queue import LLMRequestCancelled, get_llm_request_queue
from app.services.request_context import get_request_context
from app.services.model_registry import ModelRegistry, ModelSelection, VisionModelPool
from app.services.model_health import get_model_health_tracker

logger = logging.getLogger(__name__)
_RETRYABLE_HTTP_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class VisionImage:
    page: int
    data: bytes
    mime_type: str


class DashScopeVisionGateway:
    """Call a Bailian Qwen-VL compatible endpoint with in-memory images."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_health = get_model_health_tracker(
            failure_threshold=getattr(settings, "llm_model_failure_threshold", 2),
            cooldown_seconds=getattr(settings, "llm_model_cooldown_seconds", 60),
        )

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Sequence[VisionImage],
        *,
        model: str | None = None,
        pool: VisionModelPool = "ocr",
    ) -> dict[str, Any]:
        result, _ = self.generate_json_with_model(
            system_prompt,
            user_prompt,
            images,
            model=model,
            pool=pool,
        )
        return result

    def generate_json_with_model(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Sequence[VisionImage],
        *,
        model: str | None = None,
        pool: VisionModelPool = "ocr",
    ) -> tuple[dict[str, Any], str]:
        if not self.settings.qwen_configured:
            raise LLMGatewayError("未配置 DASHSCOPE_API_KEY，无法调用百炼视觉模型。")
        selection = (
            ModelSelection(f"vision:explicit:{model}", (model,))
            if model
            else ModelRegistry(self.settings).vision(pool)
        )
        content: list[dict[str, Any]] = []
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
        started = time.perf_counter()
        upstream_started: float | None = None
        queue_wait_ms = 0.0
        failover_count = 0
        remaining_models = list(self.model_health.available(selection.candidates))
        context = get_request_context()
        queue = get_llm_request_queue(
            max_concurrency=getattr(self.settings, "llm_max_concurrency", 12),
            max_requests_per_minute=getattr(
                self.settings, "llm_max_requests_per_minute", 180
            ),
            model_max_concurrency=getattr(
                self.settings, "llm_model_max_concurrency", 1
            ),
        )
        try:
            while remaining_models:
                with queue.admission(
                    request_id=context.request_id,
                    user_id=context.user_id,
                    candidate_models=tuple(remaining_models),
                    pool=selection.pool,
                ) as ticket:
                    upstream_started = time.perf_counter()
                    selected_model = ticket.selected_model or remaining_models[0]
                    if ticket.started_at is not None:
                        queue_wait_ms += max(
                            0.0, (ticket.started_at - ticket.enqueued_at) * 1000
                        )
                    prompt = user_prompt
                    messages: list[dict[str, Any]]
                    if ModelRegistry.is_ocr_only(selected_model):
                        prompt = f"{system_prompt}\n\n{user_prompt}"
                        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, *content]}]
                    else:
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": [{"type": "text", "text": prompt}, *content]},
                        ]
                    payload = {
                        "model": selected_model,
                        "messages": messages,
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
                    try:
                        with urllib.request.urlopen(
                            request, timeout=self.settings.request_timeout_seconds
                        ) as response:
                            raw_body = response.read().decode("utf-8")
                    except urllib.error.HTTPError as exc:
                        body = exc.read().decode("utf-8", errors="replace")[:500]
                        if exc.code in _RETRYABLE_HTTP_STATUSES:
                            self.model_health.record_failure(selected_model)
                        if exc.code in _RETRYABLE_HTTP_STATUSES and len(remaining_models) > 1:
                            failover_count += 1
                            remaining_models.remove(selected_model)
                            self._log_failover(selection.pool, selected_model, exc.code)
                            continue
                        raise LLMGatewayError(
                            f"百炼视觉请求失败（HTTP {exc.code}）：{body}"
                        ) from exc
                    except TimeoutError as exc:
                        self.model_health.record_failure(selected_model)
                        if len(remaining_models) > 1:
                            failover_count += 1
                            remaining_models.remove(selected_model)
                            self._log_failover(selection.pool, selected_model, "timeout")
                            continue
                        raise LLMGatewayTimeoutError("百炼视觉响应超时。") from exc
                    except urllib.error.URLError as exc:
                        self.model_health.record_failure(selected_model)
                        if len(remaining_models) > 1:
                            failover_count += 1
                            remaining_models.remove(selected_model)
                            self._log_failover(selection.pool, selected_model, "connection")
                            continue
                        raise LLMGatewayError(f"百炼视觉请求无法连接：{exc}") from exc
                    self.model_health.record_success(selected_model)
                    break
                break
        except LLMRequestCancelled as exc:
            raise LLMGatewayCancelledError(str(exc)) from exc
        try:
            response_payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise LLMGatewayError("百炼视觉返回的响应不是合法 JSON。") from exc
        usage = response_payload.get("usage") if isinstance(response_payload, dict) else None
        record_model_call(
            getattr(self.settings, "profile_db_path", "profile.db"),
            service="qwen-vl",
            model=selected_model,
            duration_ms=(time.perf_counter() - (upstream_started or started)) * 1000,
            input_tokens=_usage_int(usage, "prompt_tokens", "input_tokens"),
            output_tokens=_usage_int(usage, "completion_tokens", "output_tokens"),
            metadata={
                "endpoint": "vision",
                "pool": selection.pool,
                "failover_count": failover_count,
                "images": len(images),
                "queue_wait_ms": round(queue_wait_ms, 3),
            },
        )
        content_text = DashScopeQwenGateway._extract_content(response_payload)
        return DashScopeQwenGateway._parse_json_content(content_text), selected_model

    @staticmethod
    def _log_failover(pool: str, model: str, reason: object) -> None:
        logger.warning(
            "vision model failover pool=%s model=%s reason=%s",
            pool,
            model,
            reason,
        )


def _usage_int(usage: Any, *keys: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None
