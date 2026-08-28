"""Text-only TrialAgent dataset preparation helpers.

The training pipeline deliberately stays separate from the runtime multimodal
extractor. Images are converted to server-owned text evidence before a
TrialAgent evaluation is prepared for SFT or DPO.
"""

from app.training.cases import TrialCaseInput, case_fingerprint, load_case_inputs
from app.training.export import export_dpo_records, export_sft_records
from app.training.generation import (
    CaseGenerator,
    GeneratedCaseResponse,
    build_case_generation_prompt,
    generation_fingerprint,
)
from app.training.teacher import TeacherCache, TeacherLabeler
from app.training.validation import ValidationResult, validate_evaluation

__all__ = [
    "TrialCaseInput",
    "case_fingerprint",
    "load_case_inputs",
    "export_dpo_records",
    "export_sft_records",
    "CaseGenerator",
    "GeneratedCaseResponse",
    "build_case_generation_prompt",
    "generation_fingerprint",
    "TeacherCache",
    "TeacherLabeler",
    "ValidationResult",
    "validate_evaluation",
]
