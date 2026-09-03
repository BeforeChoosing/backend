"""Configuration-backed model pools used by text and multimodal gateways."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.config import Settings


TextModelTier = Literal["fast", "balanced", "reasoning"]
VisionModelPool = Literal["fast", "ocr", "general", "reasoning"]


@dataclass(frozen=True)
class ModelSelection:
    pool: str
    candidates: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError(f"模型池 {self.pool} 不能为空。")


class ModelRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings

    def text(self, tier: TextModelTier | None = None) -> ModelSelection:
        if tier == "fast":
            candidates = self.settings.qwen_fast_models
        elif tier == "reasoning":
            candidates = self.settings.qwen_reasoning_models
        elif tier == "balanced":
            candidates = self.settings.qwen_balanced_models
        else:
            candidates = (
                *self.settings.qwen_balanced_models,
                *self.settings.qwen_reasoning_models,
            )
        return ModelSelection(f"text:{tier or 'core'}", _unique(candidates))

    def vision(self, pool: VisionModelPool) -> ModelSelection:
        candidates = {
            "fast": self.settings.ocr_fast_models,
            "ocr": self.settings.ocr_specialist_models,
            "general": self.settings.vision_general_models,
            "reasoning": self.settings.vision_reasoning_models,
        }[pool]
        return ModelSelection(f"vision:{pool}", _unique(candidates))

    @staticmethod
    def is_ocr_only(model: str) -> bool:
        return model == "qwen3.5-ocr" or model.startswith("qwen-vl-ocr")


def _unique(models: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(model.strip() for model in models if model.strip()))
