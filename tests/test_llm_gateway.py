import json
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
