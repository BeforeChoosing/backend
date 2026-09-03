from types import SimpleNamespace

import pytest

from app.services.bailian_retrieval import (
    DashScopeEmbeddingGateway,
    DashScopeRerankGateway,
    EmbeddingGatewayError,
)


class _FakeResponse:
    def __init__(self, payload: dict):
        import json

        self.body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def read(self):
        return self.body


def _settings(**overrides):
    values = {
        "dashscope_api_key": "test-key",
        "bailian_embedding_url": "https://example.com/embedding",
        "bailian_embedding_model": "qwen3.7-text-embedding",
        "bailian_embedding_dimension": 3,
        "bailian_embedding_batch_size": 20,
        "bailian_rerank_url": "https://example.com/rerank",
        "bailian_rerank_model": "gte-rerank-v2",
        "request_timeout_seconds": 4,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_embedding_gateway_parses_direct_response_in_input_order(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        import json

        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "output": {
                    "embeddings": [
                        {"text_index": 1, "embedding": [0.0, 1.0, 0.0]},
                        {"text_index": 0, "embedding": [1.0, 0.0, 0.0]},
                    ]
                }
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    vectors = DashScopeEmbeddingGateway(_settings()).embed(
        ["第一段", "第二段"], text_type="document"
    )

    assert vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert captured["payload"]["model"] == "qwen3.7-text-embedding"
    assert captured["payload"]["input"] == {"texts": ["第一段", "第二段"]}
    assert captured["payload"]["parameters"] == {
        "dimension": 3,
        "output_type": "dense",
        "text_type": "document",
    }
    assert captured["timeout"] == 4


def test_rerank_gateway_parses_scores_and_sorts(monkeypatch):
    def fake_urlopen(request, timeout):
        return _FakeResponse(
            {
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.82},
                        {"index": 0, "relevance_score": 0.91},
                    ]
                }
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    hits = DashScopeRerankGateway(_settings()).rerank(
        "如何验证用户需求", ["段落 A", "段落 B"], top_n=2
    )

    assert [(hit.index, hit.relevance_score) for hit in hits] == [
        (0, 0.91),
        (1, 0.82),
    ]


def test_embedding_gateway_requires_api_key():
    gateway = DashScopeEmbeddingGateway(_settings(dashscope_api_key=""))

    with pytest.raises(EmbeddingGatewayError, match="DASHSCOPE_API_KEY"):
        gateway.embed(["需要密钥"], text_type="query")
