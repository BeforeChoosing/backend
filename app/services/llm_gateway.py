import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from app.config import Settings
from app.services.audit_log import record_model_call
from app.services.llm_request_queue import (
    LLMRequestCancelled,
    get_llm_request_queue,
)
from app.services.request_context import get_request_context
from app.services.model_registry import ModelRegistry, ModelSelection, TextModelTier


class LLMGatewayError(RuntimeError):
    """Raised when the configured Qwen gateway cannot produce a response."""


class LLMGatewayTimeoutError(LLMGatewayError):
    """Raised when an admitted DashScope request exceeds its upstream timeout."""


class LLMGatewayCancelledError(LLMGatewayError):
    """Raised after the user explicitly cancels their queued request."""


def llm_error_status(error: LLMGatewayError) -> int:
    if isinstance(error, LLMGatewayCancelledError):
        return 499
    if isinstance(error, LLMGatewayTimeoutError):
        return 504
    return 503


class DashScopeQwenGateway:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str | None = None,
        tier: TextModelTier | None = None,
    ) -> dict[str, Any]:
        if not self.settings.qwen_configured:
            raise LLMGatewayError(
                "未配置 DASHSCOPE_API_KEY。请在 backend/.env 中配置百炼密钥，"
                "不会使用伪造结果代替模型响应。"
            )

        context = get_request_context()
        selection = self._selection(model=model, tier=tier)
        queue = get_llm_request_queue(
            max_concurrency=getattr(self.settings, "llm_max_concurrency", 2),
            max_requests_per_minute=getattr(
                self.settings, "llm_max_requests_per_minute", 30
            ),
            model_max_concurrency=getattr(
                self.settings, "llm_model_max_concurrency", 1
            ),
        )
        # Admission happens before the upstream deadline starts. Queue wait is
        # tracked separately so a busy FIFO does not consume the DashScope
        # response timeout.
        started = time.perf_counter()
        upstream_started: float | None = None
        queue_wait_ms = 0.0
        try:
            with queue.admission(
                request_id=context.request_id,
                user_id=context.user_id,
                candidate_models=selection.candidates,
                pool=selection.pool,
            ) as ticket:
                upstream_started = time.perf_counter()
                selected_model = ticket.selected_model or selection.candidates[0]
                if ticket.started_at is not None:
                    queue_wait_ms = max(0.0, (ticket.started_at - ticket.enqueued_at) * 1000)
                request = self._request(
                    model=selected_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                try:
                    with urllib.request.urlopen(
                        request, timeout=self.settings.request_timeout_seconds
                    ) as response:
                        raw_body = response.read().decode("utf-8")
                except urllib.error.HTTPError as exc:
                    body = exc.read().decode("utf-8", errors="replace")[:500]
                    raise LLMGatewayError(f"百炼请求失败（HTTP {exc.code}）：{body}") from exc
                except TimeoutError as exc:
                    raise LLMGatewayTimeoutError("百炼响应超时，请稍后查看当前记录。") from exc
                except urllib.error.URLError as exc:
                    if isinstance(exc.reason, TimeoutError):
                        raise LLMGatewayTimeoutError(
                            "百炼响应超时，请稍后查看当前记录。"
                        ) from exc
                    raise LLMGatewayError(f"百炼请求无法连接：{exc}") from exc
        except LLMRequestCancelled as exc:
            raise LLMGatewayCancelledError(str(exc)) from exc

        try:
            response_payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise LLMGatewayError("百炼返回的响应不是合法 JSON。") from exc

        usage = response_payload.get("usage") if isinstance(response_payload, dict) else None
        record_model_call(
            getattr(self.settings, "profile_db_path", "profile.db"),
            service="qwen",
            model=selected_model,
            duration_ms=(time.perf_counter() - (upstream_started or started)) * 1000,
            input_tokens=_usage_int(usage, "prompt_tokens", "input_tokens"),
            output_tokens=_usage_int(usage, "completion_tokens", "output_tokens"),
            metadata={
                "endpoint": "chat",
                "pool": selection.pool,
                "queue_wait_ms": round(queue_wait_ms, 3),
            },
        )
        content = self._extract_content(response_payload)
        return self._parse_json_content(content)

    def stream_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        on_delta: Callable[[str], None],
        model: str | None = None,
        tier: TextModelTier | None = None,
    ) -> dict[str, Any]:
        """Stream OpenAI-compatible content deltas and return validated JSON.

        The callback receives model text as soon as DashScope emits it. The
        complete response is still parsed at the end so callers retain the
        same structured-output guarantees as ``generate_json``.
        """

        if not self.settings.qwen_configured:
            raise LLMGatewayError(
                "未配置 DASHSCOPE_API_KEY。请在 backend/.env 中配置百炼密钥，"
                "不会使用伪造结果代替模型响应。"
            )

        context = get_request_context()
        selection = self._selection(model=model, tier=tier)
        queue = get_llm_request_queue(
            max_concurrency=getattr(self.settings, "llm_max_concurrency", 2),
            max_requests_per_minute=getattr(
                self.settings, "llm_max_requests_per_minute", 30
            ),
            model_max_concurrency=getattr(
                self.settings, "llm_model_max_concurrency", 1
            ),
        )
        started = time.perf_counter()
        upstream_started: float | None = None
        queue_wait_ms = 0.0
        content_parts: list[str] = []
        usage: Any = None
        try:
            with queue.admission(
                request_id=context.request_id,
                user_id=context.user_id,
                candidate_models=selection.candidates,
                pool=selection.pool,
            ) as ticket:
                upstream_started = time.perf_counter()
                selected_model = ticket.selected_model or selection.candidates[0]
                if ticket.started_at is not None:
                    queue_wait_ms = max(
                        0.0, (ticket.started_at - ticket.enqueued_at) * 1000
                    )
                request = self._request(
                    model=selected_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    stream=True,
                )
                try:
                    with urllib.request.urlopen(
                        request, timeout=self.settings.request_timeout_seconds
                    ) as response:
                        for raw_line in response:
                            if ticket.cancel_requested:
                                raise LLMRequestCancelled("请求已由用户取消。")
                            line = raw_line.decode("utf-8", errors="replace").strip()
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if not data or data == "[DONE]":
                                continue
                            try:
                                chunk = json.loads(data)
                            except json.JSONDecodeError as exc:
                                raise LLMGatewayError("百炼流式响应包不是合法 JSON。") from exc
                            if isinstance(chunk, dict) and chunk.get("usage"):
                                usage = chunk["usage"]
                            delta = self._extract_stream_delta(chunk)
                            if delta:
                                content_parts.append(delta)
                                on_delta(delta)
                except urllib.error.HTTPError as exc:
                    body = exc.read().decode("utf-8", errors="replace")[:500]
                    raise LLMGatewayError(
                        f"百炼请求失败（HTTP {exc.code}）：{body}"
                    ) from exc
                except TimeoutError as exc:
                    raise LLMGatewayTimeoutError("百炼响应超时，请稍后查看当前记录。") from exc
                except urllib.error.URLError as exc:
                    if isinstance(exc.reason, TimeoutError):
                        raise LLMGatewayTimeoutError(
                            "百炼响应超时，请稍后查看当前记录。"
                        ) from exc
                    raise LLMGatewayError(f"百炼请求无法连接：{exc}") from exc
        except LLMRequestCancelled as exc:
            raise LLMGatewayCancelledError(str(exc)) from exc

        record_model_call(
            getattr(self.settings, "profile_db_path", "profile.db"),
            service="qwen",
            model=selected_model,
            duration_ms=(time.perf_counter() - (upstream_started or started)) * 1000,
            input_tokens=_usage_int(usage, "prompt_tokens", "input_tokens"),
            output_tokens=_usage_int(usage, "completion_tokens", "output_tokens"),
            metadata={
                "endpoint": "chat-stream",
                "pool": selection.pool,
                "queue_wait_ms": round(queue_wait_ms, 3),
            },
        )
        content = "".join(content_parts)
        if not content.strip():
            raise LLMGatewayError("百炼流式响应中没有可读取的模型文本。")
        return self._parse_json_content(content)

    def _selection(
        self,
        *,
        model: str | None,
        tier: TextModelTier | None,
    ) -> ModelSelection:
        if model:
            return ModelSelection(f"text:explicit:{model}", (model,))
        return ModelRegistry(self.settings).text(tier)

    def _request(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        stream: bool = False,
    ) -> urllib.request.Request:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        headers = {
            "Authorization": f"Bearer {self.settings.dashscope_api_key}",
            "Content-Type": "application/json",
        }
        if stream:
            headers["Accept"] = "text/event-stream"
        return urllib.request.Request(
            self.settings.dashscope_base_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )

    @staticmethod
    def _extract_stream_delta(payload: Any) -> str:
        if not isinstance(payload, dict):
            return ""
        choices = payload.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return ""
        delta = choices[0].get("delta") or {}
        content = delta.get("content") if isinstance(delta, dict) else None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            )
        return ""

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
