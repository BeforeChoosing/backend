from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.agents.trial_agent import TrialAgent
from app.config import Settings, get_settings
from app.evaluation.dataset import dataset_sha256
from app.evaluation.models import EvaluationCase, EvaluationReport, TrialArm
from app.evaluation.metrics import evaluate_case
from app.evaluation.report import build_report
from app.schemas.trial import TrialEvaluation
from app.services.llm_gateway import DashScopeQwenGateway
from app.services.trial_scoring import TrialScoringService
from app.services.trial_verification import TrialVerificationService
from app.tasks.catalog import get_task_definition


class PredictionRecord(BaseModel):
    case_id: str = Field(min_length=1, max_length=120)
    arm: TrialArm
    evaluation: dict[str, Any]
    valid_evidence_refs: list[str] = Field(default_factory=list, max_length=80)
    api_calls: int = Field(default=1, ge=0)
    verifier_api_calls: int = Field(default=0, ge=0)
    verifier_triggered: bool = False
    latency_ms: float | None = Field(default=None, ge=0)


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            cases.append(EvaluationCase.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise ValueError(f"无法读取评测集 {path} 第 {line_number} 行：{exc}") from exc
    if not cases:
        raise ValueError(f"评测集为空：{path}")
    return cases


def load_prediction_records(path: str | Path) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(PredictionRecord.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise ValueError(f"无法读取预测结果 {path} 第 {line_number} 行：{exc}") from exc
    if not records:
        raise ValueError(f"预测结果为空：{path}")
    return records


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def run_offline_evaluation(
    cases_path: str | Path,
    predictions: list[PredictionRecord],
) -> EvaluationReport:
    cases = {case.case_id: case for case in load_evaluation_cases(cases_path)}
    grouped: dict[str, list] = {}
    seen: set[tuple[str, str]] = set()
    for record in predictions:
        case = cases.get(record.case_id)
        if case is None:
            raise ValueError(f"预测结果引用了评测集之外的 case_id：{record.case_id}")
        key = (record.case_id, record.arm)
        if key in seen:
            raise ValueError(f"预测结果重复：{record.arm}/{record.case_id}")
        seen.add(key)
        result = evaluate_case(
            case,
            record.evaluation,
            valid_evidence_refs=record.valid_evidence_refs,
            api_calls=record.api_calls,
            verifier_api_calls=record.verifier_api_calls,
            verifier_triggered=record.verifier_triggered,
            latency_ms=record.latency_ms,
        )
        grouped.setdefault(record.arm, []).append(result)
    return build_report(
        dataset_version="trial-agent-eval-v1",
        dataset_sha256=dataset_sha256(cases_path),
        model_id="offline-predictions",
        prompt_version="supplied-predictions",
        cases_by_arm=grouped,
        metadata={"git_commit": _git_commit(), "mode": "offline"},
    )


def _arm_config(arm: TrialArm, settings: Settings) -> tuple[str, str, bool]:
    if arm == "base_qwen":
        return "base", settings.trial_base_model or settings.qwen_model, False
    if arm == "prompt_hardened":
        return "prompt", settings.qwen_model, False
    if not settings.trial_sft_model.strip():
        raise ValueError("SFT 对照需要先设置 TRIAL_SFT_MODEL")
    return "prompt", settings.trial_sft_model, arm == "sft_validator"


async def _run_live_case(
    case: EvaluationCase,
    arm: TrialArm,
    settings: Settings,
) -> tuple[dict[str, Any], list[str], bool, float]:
    task = get_task_definition(case.task_id)
    evidence_bundle = TrialScoringService.build_evidence(task, case.answer, [])
    prompt_variant, model_id, with_validator = _arm_config(arm, settings)
    agent = TrialAgent(
        DashScopeQwenGateway(settings),
        prompt_variant=prompt_variant,
        model_override=model_id,
    )
    started = time.perf_counter()
    evaluation = await agent.evaluate_dynamic(task, case.answer, [], evidence_bundle)
    evaluation, evidence_bundle = TrialScoringService.finalize_dynamic(
        task,
        case.answer,
        [],
        evaluation,
    )
    verifier_triggered = False
    if with_validator:
        verification = TrialVerificationService(
            min_evidence_coverage=settings.trial_verifier_min_evidence_coverage
        ).check(task, case.answer, evidence_bundle, evaluation)
        evaluation = TrialVerificationService.attach(evaluation, verification)
        verifier_triggered = verification.triggered
    latency_ms = (time.perf_counter() - started) * 1000
    return (
        evaluation.model_dump(mode="json"),
        [item.id for item in evidence_bundle.items],
        verifier_triggered,
        latency_ms,
    )


async def run_live_evaluation(
    cases_path: str | Path,
    *,
    settings: Settings | None = None,
    arms: tuple[TrialArm, ...] = ("base_qwen", "prompt_hardened", "sft", "sft_validator"),
) -> EvaluationReport:
    settings = settings or get_settings()
    cases = load_evaluation_cases(cases_path)
    grouped: dict[str, list] = {}
    for arm in arms:
        for case in cases:
            evaluation, valid_refs, verifier_triggered, latency_ms = await _run_live_case(
                case,
                arm,
                settings,
            )
            result = evaluate_case(
                case,
                evaluation,
                valid_evidence_refs=valid_refs,
                api_calls=1,
                verifier_api_calls=0,
                verifier_triggered=verifier_triggered,
                latency_ms=latency_ms,
            )
            grouped.setdefault(arm, []).append(result)
    return build_report(
        dataset_version="trial-agent-eval-v1",
        dataset_sha256=dataset_sha256(cases_path),
        model_id=settings.qwen_model,
        prompt_version=TrialAgent.PROMPT_VERSION,
        cases_by_arm=grouped,
        metadata={
            "git_commit": _git_commit(),
            "mode": "live",
            "sft_model": settings.trial_sft_model or None,
            "verifier_policy": "only sft_validator runs the deterministic gate",
        },
    )
