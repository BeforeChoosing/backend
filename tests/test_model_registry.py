from app.config import Settings
from app.services.model_registry import ModelRegistry


def test_registry_exposes_configured_text_tiers_and_vision_pools() -> None:
    settings = Settings(
        qwen_fast_models=("fast-a", "fast-b"),
        qwen_balanced_models=("balanced-a",),
        qwen_reasoning_models=("reasoning-a",),
        qwen_comprehensive_models=("comprehensive-a",),
        qwen_thinking_models=("thinking-a", "thinking-b"),
        qwen_thinking_fallback_models=("thinking-fallback",),
        ocr_specialist_models=("ocr-a", "ocr-b"),
        vision_general_models=("vision-a",),
    )
    registry = ModelRegistry(settings)

    assert registry.text("fast").candidates == ("fast-a", "fast-b")
    assert registry.text("comprehensive").candidates == ("comprehensive-a",)
    assert registry.text("thinking").candidates == (
        "thinking-a", "thinking-b", "thinking-fallback"
    )
    assert registry.is_thinking_model("thinking-a") is True
    assert registry.is_thinking_model("custom-thinking-model") is True
    assert registry.text().candidates == ("balanced-a", "reasoning-a")
    assert registry.vision("ocr").candidates == ("ocr-a", "ocr-b")
    assert registry.vision("general").candidates == ("vision-a",)


def test_registry_identifies_ocr_models_without_system_message_support() -> None:
    assert ModelRegistry.is_ocr_only("qwen3.5-ocr") is True
    assert ModelRegistry.is_ocr_only("qwen-vl-ocr-2025-11-20") is True
    assert ModelRegistry.is_ocr_only("qwen3-vl-plus") is False
