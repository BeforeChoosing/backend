from app.config import Settings
from app.services.model_registry import ModelRegistry


def test_registry_exposes_configured_text_tiers_and_vision_pools() -> None:
    settings = Settings(
        qwen_fast_models=("fast-a", "fast-b"),
        qwen_balanced_models=("balanced-a",),
        qwen_reasoning_models=("reasoning-a",),
        ocr_specialist_models=("ocr-a", "ocr-b"),
        vision_general_models=("vision-a",),
    )
    registry = ModelRegistry(settings)

    assert registry.text("fast").candidates == ("fast-a", "fast-b")
    assert registry.text().candidates == ("balanced-a", "reasoning-a")
    assert registry.vision("ocr").candidates == ("ocr-a", "ocr-b")
    assert registry.vision("general").candidates == ("vision-a",)


def test_registry_identifies_ocr_models_without_system_message_support() -> None:
    assert ModelRegistry.is_ocr_only("qwen3.5-ocr") is True
    assert ModelRegistry.is_ocr_only("qwen-vl-ocr-2025-11-20") is True
    assert ModelRegistry.is_ocr_only("qwen3-vl-plus") is False
