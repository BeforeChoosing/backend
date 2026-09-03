import json
import logging
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
from app.services.model_health import get_model_health_tracker

logger = logging.getLogger(__name__)
_RETRYABLE_HTTP_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


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
        self.model_health = get_model_health_tracker(
            failure_threshold=getattr(settings, "llm_model_failure_threshold", 2),
            cooldown_seconds=getattr(settings, "llm_model_cooldown_seconds", 60),
        )

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str | None = None,
        tier: TextModelTier | None = None,
        validator: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if not self.settings.qwen_configured:
            raise LLMGatewayError(
                "未配置 DASHSCOPE_API_KEY。请在 backend/.env 中配置百炼密钥，"
                "不会使用伪造结果代替模型响应。"
            )

        # Dedicated thinking models do not support DashScope JSON mode.  Use
        # the streaming endpoint internally, collect the complete text, and
        # run the same validation/repair path as normal structured calls.
        if self._thinking_requested(model=model, tier=tier):
            return self.stream_json(
                system_prompt,
                user_prompt,
                on_delta=lambda _delta: None,
                on_thinking_delta=lambda _delta: None,
                model=model,
                tier=tier,
                validator=validator,
            )

        context = get_request_context()
        selection = self._selection(model=model, tier=tier)
        queue = get_llm_request_queue(
            max_concurrency=getattr(self.settings, "llm_max_concurrency", 12),
            max_requests_per_minute=getattr(
                self.settings, "llm_max_requests_per_minute", 180
            ),
            model_max_concurrency=getattr(
                self.settings, "llm_model_max_concurrency", 1
            ),
        )
        # Admission happens before the upstream deadline starts. Queue wait is
        # tracked separately so a busy FIFO does not consume the DashScope
        # response timeout.
        started = time.perf_counter()
        queue_wait_ms = 0.0
        failover_count = 0
        remaining_models = list(self.model_health.available(selection.candidates))
        last_error: LLMGatewayError | None = None
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
                        last_error = LLMGatewayError(
                            f"百炼请求失败（HTTP {exc.code}）：{body}"
                        )
                        if exc.code in _RETRYABLE_HTTP_STATUSES:
                            self.model_health.record_failure(selected_model)
                        if exc.code in _RETRYABLE_HTTP_STATUSES and len(remaining_models) > 1:
                            failover_count += 1
                            remaining_models.remove(selected_model)
                            self._log_failover(selection.pool, selected_model, exc.code)
                            continue
                        raise last_error from exc
                    except TimeoutError as exc:
                        self.model_health.record_failure(selected_model)
                        if len(remaining_models) > 1:
                            failover_count += 1
                            remaining_models.remove(selected_model)
                            self._log_failover(selection.pool, selected_model, "timeout")
                            continue
                        raise LLMGatewayTimeoutError(
                            "百炼响应超时，请稍后查看当前记录。"
                        ) from exc
                    except urllib.error.URLError as exc:
                        self.model_health.record_failure(selected_model)
                        if len(remaining_models) > 1:
                            failover_count += 1
                            remaining_models.remove(selected_model)
                            self._log_failover(selection.pool, selected_model, "connection")
                            continue
                        if isinstance(exc.reason, TimeoutError):
                            raise LLMGatewayTimeoutError(
                                "百炼响应超时，请稍后查看当前记录。"
                            ) from exc
                        raise LLMGatewayError(f"百炼请求无法连接：{exc}") from exc

                    try:
                        response_payload = json.loads(raw_body)
                        if not isinstance(response_payload, dict):
                            raise LLMGatewayError("百炼返回的响应不是 JSON 对象。")
                        content = self._extract_content(response_payload)
                        parsed = self._parse_json_content(content)
                        if validator is not None:
                            validator(parsed)
                    except (json.JSONDecodeError, LLMGatewayError, TypeError, ValueError) as exc:
                        last_error = (
                            exc
                            if isinstance(exc, LLMGatewayError)
                            else LLMGatewayError(f"模型输出未通过结构校验：{exc}")
                        )
                        self.model_health.record_failure(selected_model)
                        if len(remaining_models) > 1:
                            failover_count += 1
                            remaining_models.remove(selected_model)
                            self._log_failover(
                                selection.pool,
                                selected_model,
                                type(exc).__name__,
                            )
                            continue
                        raise last_error from exc

                    self.model_health.record_success(selected_model)
                    usage = response_payload.get("usage")
                    record_model_call(
                        getattr(self.settings, "profile_db_path", "profile.db"),
                        service="qwen",
                        model=selected_model,
                        duration_ms=(time.perf_counter() - upstream_started) * 1000,
                        input_tokens=_usage_int(usage, "prompt_tokens", "input_tokens"),
                        output_tokens=_usage_int(usage, "completion_tokens", "output_tokens"),
                        metadata={
                            "endpoint": "chat",
                            "pool": selection.pool,
                            "failover_count": failover_count,
                            "queue_wait_ms": round(queue_wait_ms, 3),
                        },
                    )
                    parsed["_selected_model"] = selected_model
                    parsed["_model_pool"] = selection.pool
                    return parsed
        except LLMRequestCancelled as exc:
            raise LLMGatewayCancelledError(str(exc)) from exc
        raise last_error or LLMGatewayError("当前模型池没有可用模型。")

    @staticmethod
    def _log_failover(pool: str, model: str, reason: object) -> None:
        logger.warning(
            "model failover pool=%s model=%s reason=%s",
            pool,
            model,
            reason,
        )

    def stream_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        on_delta: Callable[[str], None],
        on_reset: Callable[[], None] | None = None,
        on_thinking_delta: Callable[[str], None] | None = None,
        model: str | None = None,
        tier: TextModelTier | None = None,
        validator: Callable[[dict[str, Any]], None] | None = None,
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
        registry = ModelRegistry(self.settings)
        selection = self._selection(model=model, tier=tier)
        queue = get_llm_request_queue(
            max_concurrency=getattr(self.settings, "llm_max_concurrency", 12),
            max_requests_per_minute=getattr(
                self.settings, "llm_max_requests_per_minute", 180
            ),
            model_max_concurrency=getattr(
                self.settings, "llm_model_max_concurrency", 1
            ),
        )
        queue_wait_ms = 0.0
        failover_count = 0
        remaining_models = list(self.model_health.available(selection.candidates))
        last_error: LLMGatewayError | None = None
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
                    thinking_enabled = registry.is_thinking_model(selected_model)
                    if ticket.started_at is not None:
                        queue_wait_ms += max(
                            0.0, (ticket.started_at - ticket.enqueued_at) * 1000
                        )
                    request = self._request(
                        model=selected_model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        stream=True,
                        thinking_enabled=thinking_enabled,
                        preserve_thinking=(
                            thinking_enabled
                            and getattr(self.settings, "thinking_preserve_history", False)
                        ),
                    )
                    content_parts: list[str] = []
                    reasoning_parts: list[str] = []
                    usage: Any = None
                    emitted = False
                    try:
                        with urllib.request.urlopen(
                            request,
                            timeout=(
                                getattr(
                                    self.settings,
                                    "thinking_request_timeout_seconds",
                                    self.settings.request_timeout_seconds,
                                )
                                if thinking_enabled
                                else self.settings.request_timeout_seconds
                            ),
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
                                    raise LLMGatewayError(
                                        "百炼流式响应包不是合法 JSON。"
                                    ) from exc
                                if isinstance(chunk, dict) and chunk.get("usage"):
                                    usage = chunk["usage"]
                                reasoning_delta = self._extract_stream_reasoning_delta(chunk)
                                if reasoning_delta:
                                    reasoning_parts.append(reasoning_delta)
                                    if on_thinking_delta is not None:
                                        on_thinking_delta(reasoning_delta)
                                    emitted = True
                                delta = self._extract_stream_delta(chunk)
                                if delta:
                                    content_parts.append(delta)
                                    on_delta(delta)
                                    emitted = True
                    except urllib.error.HTTPError as exc:
                        body = exc.read().decode("utf-8", errors="replace")[:500]
                        last_error = LLMGatewayError(
                            f"百炼请求失败（HTTP {exc.code}）：{body}"
                        )
                        retryable = exc.code in _RETRYABLE_HTTP_STATUSES
                    except TimeoutError as exc:
                        last_error = LLMGatewayTimeoutError(
                            "百炼响应超时，请稍后查看当前记录。"
                        )
                        retryable = True
                    except urllib.error.URLError as exc:
                        if isinstance(exc.reason, TimeoutError):
                            last_error = LLMGatewayTimeoutError(
                                "百炼响应超时，请稍后查看当前记录。"
                            )
                        else:
                            last_error = LLMGatewayError(
                                f"百炼请求无法连接：{exc}"
                            )
                        retryable = True
                    else:
                        try:
                            content = "".join(content_parts)
                            if not content.strip():
                                raise LLMGatewayError(
                                    "百炼流式响应中没有可读取的模型文本。"
                                )
                            parsed = self._parse_json_content(content)
                            if validator is not None:
                                validator(parsed)
                        except (LLMGatewayError, TypeError, ValueError) as exc:
                            if thinking_enabled:
                                try:
                                    parsed = self._repair_json(
                                        system_prompt=system_prompt,
                                        raw_content=content,
                                        validator=validator,
                                    )
                                except LLMGatewayError as repair_exc:
                                    last_error = repair_exc
                                    retryable = True
                                else:
                                    self.model_health.record_success(selected_model)
                                    repair_model = parsed.pop("_repair_model", None)
                                    record_model_call(
                                        getattr(
                                            self.settings,
                                            "profile_db_path",
                                            "profile.db",
                                        ),
                                        service="qwen",
                                        model=selected_model,
                                        duration_ms=(
                                            time.perf_counter() - upstream_started
                                        )
                                        * 1000,
                                        input_tokens=_usage_int(
                                            usage, "prompt_tokens", "input_tokens"
                                        ),
                                        output_tokens=_usage_int(
                                            usage, "completion_tokens", "output_tokens"
                                        ),
                                        metadata={
                                            "endpoint": "chat-stream",
                                            "pool": selection.pool,
                                            "failover_count": failover_count,
                                            "queue_wait_ms": round(queue_wait_ms, 3),
                                            "thinking_enabled": thinking_enabled,
                                            "reasoning_tokens": _reasoning_tokens(usage),
                                            "json_repair": True,
                                            "repair_model": repair_model,
                                        },
                                    )
                                    parsed["_selected_model"] = selected_model
                                    parsed["_model_pool"] = selection.pool
                                    parsed["_thinking_enabled"] = thinking_enabled
                                    parsed["_reasoning_content"] = "".join(
                                        reasoning_parts
                                    )[:24000]
                                    parsed["_reasoning_tokens"] = _reasoning_tokens(
                                        usage
                                    )
                                    return parsed
                            else:
                                last_error = (
                                    exc
                                    if isinstance(exc, LLMGatewayError)
                                    else LLMGatewayError(
                                        f"模型输出未通过结构校验：{exc}"
                                    )
                                )
                                retryable = True
                        else:
                            self.model_health.record_success(selected_model)
                            record_model_call(
                                getattr(
                                    self.settings,
                                    "profile_db_path",
                                    "profile.db",
                                ),
                                service="qwen",
                                model=selected_model,
                                duration_ms=(
                                    time.perf_counter() - upstream_started
                                )
                                * 1000,
                                input_tokens=_usage_int(
                                    usage, "prompt_tokens", "input_tokens"
                                ),
                                output_tokens=_usage_int(
                                    usage, "completion_tokens", "output_tokens"
                                ),
                                metadata={
                                    "endpoint": "chat-stream",
                                    "pool": selection.pool,
                                    "failover_count": failover_count,
                                    "queue_wait_ms": round(queue_wait_ms, 3),
                                    "thinking_enabled": thinking_enabled,
                                    "reasoning_tokens": _reasoning_tokens(usage),
                                },
                            )
                            parsed["_selected_model"] = selected_model
                            parsed["_model_pool"] = selection.pool
                            parsed["_thinking_enabled"] = thinking_enabled
                            parsed["_reasoning_content"] = "".join(reasoning_parts)[
                                :24000
                            ]
                            parsed["_reasoning_tokens"] = _reasoning_tokens(usage)
                            return parsed

                    if retryable:
                        self.model_health.record_failure(selected_model)
                    if retryable and len(remaining_models) > 1:
                        if emitted:
                            if on_reset is None:
                                raise last_error or LLMGatewayError(
                                    "流式回复中断，无法安全切换模型。"
                                )
                            on_reset()
                        failover_count += 1
                        remaining_models.remove(selected_model)
                        self._log_failover(
                            selection.pool,
                            selected_model,
                            type(last_error).__name__ if last_error else "invalid",
                        )
                        continue
                    raise last_error or LLMGatewayError("模型响应无效。")
        except LLMRequestCancelled as exc:
            raise LLMGatewayCancelledError(str(exc)) from exc
        raise last_error or LLMGatewayError("当前模型池没有可用模型。")

    def _selection(
        self,
        *,
        model: str | None,
        tier: TextModelTier | None,
    ) -> ModelSelection:
        if model:
            return ModelSelection(f"text:explicit:{model}", (model,))
        return ModelRegistry(self.settings).text(tier)

    def _thinking_requested(
        self,
        *,
        model: str | None,
        tier: TextModelTier | None,
    ) -> bool:
        if tier == "thinking":
            return True
        if model:
            return ModelRegistry(self.settings).is_thinking_model(model)
        return False

    def _repair_json(
        self,
        *,
        system_prompt: str,
        raw_content: str,
        validator: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        """Convert a thinking model's free-form answer into a JSON object.

        Thinking-capable Qwen models cannot combine ``enable_thinking`` with
        JSON mode.  A short, separate fast-model call keeps the public API
        structured without asking the user to retry the request.
        """

        registry = ModelRegistry(self.settings)
        repair_pool = registry.text("fast")
        repair_model = repair_pool.candidates[0]
        repair_system = (
            "你是结构化输出修复器。只输出一个合法 JSON 对象，不要 Markdown、解释或代码围栏。"
        )
        repair_prompt = (
            "请把下面模型输出转换为满足原任务要求的 JSON 对象。保留可确定的信息，"
            "不要补造事实；如果字段缺失，使用空数组、空字符串或 false。\n"
            f"原任务提示：{system_prompt[:8000]}\n"
            f"模型输出：{raw_content[:24000]}"
        )
        repaired = self.generate_json(
            repair_system,
            repair_prompt,
            model=repair_model,
            tier="fast",
            validator=validator,
        )
        repaired["_repair_model"] = repair_model
        return repaired

    def _request(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        stream: bool = False,
        thinking_enabled: bool = False,
        preserve_thinking: bool = False,
    ) -> urllib.request.Request:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "enable_thinking": thinking_enabled,
        }
        if not thinking_enabled:
            payload["response_format"] = {"type": "json_object"}
        elif getattr(self.settings, "thinking_budget", 0) > 0:
            payload["thinking_budget"] = self.settings.thinking_budget
        if thinking_enabled and preserve_thinking:
            payload["preserve_thinking"] = True
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
    def _extract_stream_reasoning_delta(payload: Any) -> str:
        if not isinstance(payload, dict):
            return ""
        choices = payload.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return ""
        delta = choices[0].get("delta") or {}
        if not isinstance(delta, dict):
            return ""
        reasoning = delta.get("reasoning_content")
        if reasoning is None:
            reasoning = delta.get("reasoning")
        if isinstance(reasoning, str):
            return reasoning
        if isinstance(reasoning, list):
            return "".join(
                item.get("text", "")
                for item in reasoning
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


def _reasoning_tokens(usage: Any) -> int | None:
    if not isinstance(usage, dict):
        return None
    direct = usage.get("reasoning_tokens")
    if isinstance(direct, int) and not isinstance(direct, bool):
        return direct
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict):
        nested = details.get("reasoning_tokens")
        if isinstance(nested, int) and not isinstance(nested, bool):
            return nested
    return None
