import json
import io
import urllib.error
from contextlib import contextmanager
from time import sleep

import app.config as config
from app.config import Settings
from app.services import llm_gateway
from app.services.llm_request_queue import QueueTicket


def test_default_llm_timeout_allows_slow_structured_qwen_response() -> None:
    # The repository's local .env may intentionally override the default for
    # another test run; assert the fallback itself rather than that override.
    assert config._DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS == 180


def test_upstream_timeout_starts_after_queue_admission(monkeypatch, tmp_path) -> None:
    observed: dict[str, float] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode()

    @contextmanager
    def admission(**kwargs):
        sleep(0.02)  # Simulate time spent waiting for a prior FIFO ticket.
        yield QueueTicket(
            request_id=kwargs["request_id"],
            user_id=kwargs["user_id"],
            state="running",
            enqueued_at=100.0,
            started_at=100.02,
        )

    class FakeQueue:
        def admission(self, **kwargs):
            return admission(**kwargs)

    def fake_urlopen(request, timeout):
        observed["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(llm_gateway, "get_llm_request_queue", lambda **kwargs: FakeQueue())
    monkeypatch.setattr(llm_gateway.urllib.request, "urlopen", fake_urlopen)
    settings = Settings(
        dashscope_api_key="test-key",
        request_timeout_seconds=7,
        profile_db_path=str(tmp_path / "profile.db"),
    )

    llm_gateway.DashScopeQwenGateway(settings).generate_json("system", "user")

    assert observed["timeout"] == 7


def test_stream_json_forwards_deltas_and_uses_fast_model(monkeypatch, tmp_path) -> None:
    observed: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            chunks = [
                {"choices": [{"delta": {"content": '{"reply":"你好'}}]},
                {"choices": [{"delta": {"content": '学长"}'}}]},
                {"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 4}},
            ]
            for chunk in chunks:
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n".encode()
            yield b"data: [DONE]\n"

    @contextmanager
    def admission(**kwargs):
        yield QueueTicket(
            request_id=kwargs["request_id"],
            user_id=kwargs["user_id"],
            state="running",
            enqueued_at=100.0,
            started_at=100.0,
        )

    class FakeQueue:
        def admission(self, **kwargs):
            return admission(**kwargs)

    def fake_urlopen(request, timeout):
        observed["payload"] = json.loads(request.data.decode())
        observed["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(llm_gateway, "get_llm_request_queue", lambda **kwargs: FakeQueue())
    monkeypatch.setattr(llm_gateway.urllib.request, "urlopen", fake_urlopen)
    settings = Settings(
        dashscope_api_key="test-key",
        qwen_fast_model="qwen3.6-flash",
        request_timeout_seconds=7,
        profile_db_path=str(tmp_path / "profile.db"),
    )
    deltas: list[str] = []

    result = llm_gateway.DashScopeQwenGateway(settings).stream_json(
        "system", "user", model=settings.qwen_fast_model, on_delta=deltas.append
    )

    assert result["reply"] == "你好学长"
    assert result["_selected_model"] == "qwen3.6-flash"
    assert result["_model_pool"] == "text:explicit:qwen3.6-flash"
    assert deltas == ['{"reply":"你好', '学长"}']
    assert observed["payload"]["model"] == "qwen3.6-flash"
    assert observed["payload"]["stream"] is True
    assert observed["payload"]["enable_thinking"] is False
    assert observed["timeout"] == 7


def test_stream_json_resets_partial_output_before_model_failover(
    monkeypatch, tmp_path
) -> None:
    attempted: list[str] = []

    class FakeResponse:
        def __init__(self, content: str):
            self.content = content

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            yield (
                "data: "
                + json.dumps({"choices": [{"delta": {"content": self.content}}]})
                + "\n"
            ).encode()
            yield b"data: [DONE]\n"

    @contextmanager
    def admission(**kwargs):
        yield QueueTicket(
            request_id=kwargs["request_id"],
            user_id=kwargs["user_id"],
            state="running",
            enqueued_at=100.0,
            started_at=100.0,
        )

    class FakeQueue:
        def admission(self, **kwargs):
            return admission(**kwargs)

    def fake_urlopen(request, timeout):
        model = json.loads(request.data.decode())["model"]
        attempted.append(model)
        return FakeResponse(
            '[{"reply":"错误"}]'
            if model == "bad-stream"
            else '{"reply":"有效"}'
        )

    monkeypatch.setattr(llm_gateway, "get_llm_request_queue", lambda **kwargs: FakeQueue())
    monkeypatch.setattr(llm_gateway.urllib.request, "urlopen", fake_urlopen)
    settings = Settings(
        dashscope_api_key="test-key",
        qwen_fast_models=("bad-stream", "good-stream"),
        llm_model_failure_threshold=99,
        profile_db_path=str(tmp_path / "profile.db"),
    )
    deltas: list[str] = []
    resets: list[bool] = []

    result = llm_gateway.DashScopeQwenGateway(settings).stream_json(
        "system",
        "user",
        tier="fast",
        on_delta=deltas.append,
        on_reset=lambda: resets.append(True),
    )

    assert result["reply"] == "有效"
    assert result["_selected_model"] == "good-stream"
    assert attempted == ["bad-stream", "good-stream"]
    assert deltas == ['[{"reply":"错误"}]', '{"reply":"有效"}']
    assert resets == [True]


def test_generate_json_fails_over_to_another_model_before_returning_error(
    monkeypatch, tmp_path
) -> None:
    attempted: list[str] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode()

    @contextmanager
    def admission(**kwargs):
        yield QueueTicket(
            request_id=kwargs["request_id"],
            user_id=kwargs["user_id"],
            state="running",
            enqueued_at=100.0,
            started_at=100.0,
        )

    class FakeQueue:
        def admission(self, **kwargs):
            return admission(**kwargs)

    def fake_urlopen(request, timeout):
        model = json.loads(request.data.decode())["model"]
        attempted.append(model)
        if model == "model-unavailable":
            raise urllib.error.HTTPError(
                request.full_url,
                503,
                "unavailable",
                hdrs=None,
                fp=io.BytesIO(b"busy"),
            )
        return FakeResponse()

    monkeypatch.setattr(llm_gateway, "get_llm_request_queue", lambda **kwargs: FakeQueue())
    monkeypatch.setattr(llm_gateway.urllib.request, "urlopen", fake_urlopen)
    settings = Settings(
        dashscope_api_key="test-key",
        qwen_fast_models=("model-unavailable", "model-ready"),
        llm_model_failure_threshold=99,
        profile_db_path=str(tmp_path / "profile.db"),
    )

    result = llm_gateway.DashScopeQwenGateway(settings).generate_json(
        "system", "user", tier="fast"
    )
    assert result["_selected_model"] == "model-ready"
    assert result["_model_pool"] == "text:fast"
    assert attempted == ["model-unavailable", "model-ready"]


def test_generate_json_fails_over_when_first_model_returns_non_object_json(
    monkeypatch, tmp_path
) -> None:
    attempted: list[str] = []

    @contextmanager
    def admission(**kwargs):
        yield QueueTicket(
            request_id=kwargs["request_id"],
            user_id=kwargs["user_id"],
            state="running",
            enqueued_at=100.0,
            started_at=100.0,
        )

    class FakeQueue:
        def admission(self, **kwargs):
            return admission(**kwargs)

    class FakeResponse:
        def __init__(self, content: str):
            self.content = content

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": self.content}}]}
            ).encode()

    def fake_urlopen(request, timeout):
        model = json.loads(request.data.decode())["model"]
        attempted.append(model)
        return FakeResponse('[{"reply":"错误形状"}]' if model == "bad-shape" else '{"reply":"正常"}')

    monkeypatch.setattr(llm_gateway, "get_llm_request_queue", lambda **kwargs: FakeQueue())
    monkeypatch.setattr(llm_gateway.urllib.request, "urlopen", fake_urlopen)
    settings = Settings(
        dashscope_api_key="test-key",
        qwen_fast_models=("bad-shape", "good-shape"),
        llm_model_failure_threshold=99,
        profile_db_path=str(tmp_path / "profile.db"),
    )

    result = llm_gateway.DashScopeQwenGateway(settings).generate_json(
        "system", "user", tier="fast"
    )

    assert result["reply"] == "正常"
    assert result["_selected_model"] == "good-shape"
    assert attempted == ["bad-shape", "good-shape"]


def test_generate_json_fails_over_when_domain_validator_rejects_output(
    monkeypatch, tmp_path
) -> None:
    attempted: list[str] = []

    @contextmanager
    def admission(**kwargs):
        yield QueueTicket(
            request_id=kwargs["request_id"],
            user_id=kwargs["user_id"],
            state="running",
            enqueued_at=100.0,
            started_at=100.0,
        )

    class FakeQueue:
        def admission(self, **kwargs):
            return admission(**kwargs)

    class FakeResponse:
        def __init__(self, content: str):
            self.content = content

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": self.content}}]}
            ).encode()

    def fake_urlopen(request, timeout):
        model = json.loads(request.data.decode())["model"]
        attempted.append(model)
        content = '{"card_proposals":[]}' if model == "empty-cards" else '{"card_proposals":[{}]}'
        return FakeResponse(content)

    def require_cards(payload):
        if not payload.get("card_proposals"):
            raise ValueError("card_proposals 不能为空")

    monkeypatch.setattr(llm_gateway, "get_llm_request_queue", lambda **kwargs: FakeQueue())
    monkeypatch.setattr(llm_gateway.urllib.request, "urlopen", fake_urlopen)
    settings = Settings(
        dashscope_api_key="test-key",
        qwen_balanced_models=("empty-cards", "valid-cards"),
        llm_model_failure_threshold=99,
        profile_db_path=str(tmp_path / "profile.db"),
    )

    result = llm_gateway.DashScopeQwenGateway(settings).generate_json(
        "system", "user", tier="balanced", validator=require_cards
    )

    assert result["card_proposals"] == [{}]
    assert result["_selected_model"] == "valid-cards"
    assert attempted == ["empty-cards", "valid-cards"]
