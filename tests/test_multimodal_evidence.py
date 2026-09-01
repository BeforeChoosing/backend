import asyncio
import json

from app.config import Settings
from app.services.multimodal_evidence import MultimodalEvidenceExtractor
from app.services.vision_gateway import DashScopeVisionGateway, VisionImage


class FakeVisionGateway:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def generate_json(self, system_prompt, user_prompt, images, *, model=None):
        self.calls += 1
        assert images
        assert model == "qwen-vl-ocr"
        return self.payload


def test_image_evidence_preserves_page_and_normalized_region() -> None:
    gateway = FakeVisionGateway(
        {
            "evidence": [
                {
                    "page": 1,
                    "bbox": [100, 200, 800, 600],
                    "label": "项目结果",
                    "quote": "上线后完成 800 多笔订单流转",
                    "evidence_type": "documented_fact",
                    "confidence": 0.92,
                }
            ]
        }
    )
    extractor = MultimodalEvidenceExtractor(gateway, model="qwen-vl-ocr")

    response = asyncio.run(
        extractor.extract(
            file_name="resume.png",
            data=b"fake-image",
            mime_type="image/png",
        )
    )

    assert gateway.calls == 1
    assert response.items[0].page == 1
    assert response.items[0].bbox == [100, 200, 800, 600]
    assert response.items[0].status == "candidate"
    assert response.items[0].source_ref.startswith("material:")


def test_image_evidence_discards_invalid_regions() -> None:
    gateway = FakeVisionGateway(
        {
            "evidence": [
                {"page": 0, "bbox": [0, 0, 100, 100], "quote": "无效页"},
                {"page": 1, "bbox": [0, 0, 1200, 100], "quote": "越界"},
            ]
        }
    )
    extractor = MultimodalEvidenceExtractor(gateway, model="qwen-vl-ocr")

    response = asyncio.run(
        extractor.extract(
            file_name="screen.png",
            data=b"fake-image",
            mime_type="image/png",
        )
    )

    assert response.items == []
    assert response.rejected_count == 2


def test_vision_gateway_sends_data_url_to_bailian(monkeypatch) -> None:
    captured = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": '{"evidence": []}'}}]}
            ).encode()

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("app.services.vision_gateway.urllib.request.urlopen", fake_urlopen)
    settings = Settings(
        dashscope_api_key="test-key",
        dashscope_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        bailian_vision_model="qwen-vl-ocr",
    )
    payload = DashScopeVisionGateway(settings).generate_json(
        "system",
        "extract",
        [VisionImage(page=1, data=b"png", mime_type="image/png")],
    )

    assert payload == {"evidence": []}
    assert captured["payload"]["model"] in {
        "qwen3.5-ocr",
        "qwen-vl-ocr-2025-11-20",
    }
    assert len(captured["payload"]["messages"]) == 1
    assert captured["payload"]["messages"][0]["role"] == "user"
    assert "system" in captured["payload"]["messages"][0]["content"][0]["text"]
    assert captured["payload"]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
