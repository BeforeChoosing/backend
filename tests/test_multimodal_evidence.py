import asyncio

from app.services.multimodal_evidence import MultimodalEvidenceExtractor


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
