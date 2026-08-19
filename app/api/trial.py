import logging

from fastapi import APIRouter, HTTPException

from app.agents.trial_agent import TrialAgent
from app.config import get_settings
from app.schemas.trial import (
    A02Answer,
    A02Task,
    ObservedEvidence,
    TrialAnswerUpdateRequest,
    TrialSession,
    TrialSessionCreateRequest,
)
from app.schemas.task_catalog import (
    TrialTaskDefinition,
    TrialTaskRecommendation,
    TrialTaskRecommendationRequest,
)
from app.services.llm_gateway import DashScopeQwenGateway, LLMGatewayError
from app.services.profile_store import ProfileStore
from app.services.trial_store import TrialStore
from app.services.task_selector import recommend_trial_task
from app.tasks.a02 import A02_TASK
from app.tasks.catalog import list_task_definitions

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/trial", tags=["trial"])


def _trial_store() -> TrialStore:
    return TrialStore(get_settings().profile_db_path)


def _trial_agent() -> TrialAgent:
    return TrialAgent(DashScopeQwenGateway(get_settings()))


def _profile_store() -> ProfileStore:
    return ProfileStore(get_settings().profile_db_path)


def _get_task(task_id: str) -> A02Task:
    if task_id != A02_TASK.id:
        raise HTTPException(status_code=404, detail="当前最小 Demo 仅提供 A-02 试路任务。")
    return A02_TASK


@router.get("/catalog", response_model=list[TrialTaskDefinition])
def get_trial_catalog() -> list[TrialTaskDefinition]:
    return list_task_definitions()


@router.post("/recommendations", response_model=TrialTaskRecommendation)
def create_trial_recommendation(
    request: TrialTaskRecommendationRequest,
) -> TrialTaskRecommendation:
    selected_ids = list(dict.fromkeys(request.selected_card_ids))
    profile_store = _profile_store()
    cards = profile_store.get_cards_by_ids(selected_ids)
    if len(cards) != len(selected_ids):
        raise HTTPException(status_code=422, detail="只能使用已确认的能力卡选择试路任务。")
    try:
        return recommend_trial_task(
            cards,
            profile_store.get_completed_task_ids(),
            target_role=request.target_role,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _get_session(session_id: str) -> TrialSession:
    try:
        return _trial_store().get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="试路会话不存在。") from exc


def _validate_answer(answer: A02Answer, *, event_revealed: bool) -> None:
    case_ids = {case.id for case in A02_TASK.bad_cases}
    attributed_ids = [item.case_id for item in answer.attributions]
    if set(attributed_ids) != case_ids or len(attributed_ids) != len(case_ids):
        raise HTTPException(status_code=422, detail="请先为 8 个 Bad Case 完成归因和置信度选择。")

    if len(answer.priority_case_ids) != 2 or len(set(answer.priority_case_ids)) != 2:
        raise HTTPException(status_code=422, detail="请选出唯一的系统性 Top 2。")
    if not set(answer.priority_case_ids).issubset(case_ids):
        raise HTTPException(status_code=422, detail="Top 2 中包含未知 Case。")

    evidence_ids = {item.source_id for item in answer.evidence}
    evidence_by_id = {item.source_id: item for item in answer.evidence}
    if (
        len(answer.evidence) < 2
        or not set(answer.priority_case_ids).issubset(evidence_ids)
        or any(not evidence_by_id[case_id].explanation.strip() for case_id in answer.priority_case_ids)
    ):
        raise HTTPException(status_code=422, detail="Top 2 各至少需要一条可追溯证据。")

    plans = {item.case_id: item for item in answer.validation_plans}
    if any(
        case_id not in plans
        or not plans[case_id].action.strip()
        or not plans[case_id].expected_signal.strip()
        for case_id in answer.priority_case_ids
    ):
        raise HTTPException(status_code=422, detail="请为 Top 2 分别填写验证动作和预期信号。")

    if not event_revealed:
        raise HTTPException(status_code=422, detail="请先完成中途事件后的重新决策。")
    if (
        answer.event_decision is None
        or len(answer.event_priority_case_ids) != 2
        or len(set(answer.event_priority_case_ids)) != 2
    ):
        raise HTTPException(status_code=422, detail="请完成中途事件后的 Top 2 重排。")
    if not set(answer.event_priority_case_ids).issubset(case_ids):
        raise HTTPException(status_code=422, detail="事件后的 Top 2 中包含未知 Case。")
    if not answer.event_reason.strip():
        raise HTTPException(status_code=422, detail="请说明中途事件如何改变或确认你的判断。")


def _observed_evidence(session: TrialSession) -> ObservedEvidence:
    answer = session.answer
    return ObservedEvidence(
        task_id=session.task_id,
        statement="用户完成了 A-02 Agent Bad Case 归因、Top 2 优先级、证据引用、验证计划和事件后重排。",
        completed_steps=[
            "Bad Case 归因",
            "系统性 Top 2",
            "引用 Case 与运行指标",
            "验证计划",
            "事件后重新决策",
        ],
        evidence_refs=[item.source_id for item in answer.evidence],
        caveats=[
            "本次任务只形成 Observed Evidence，不等同于岗位胜任力认证。",
            "运行指标和案例均为模拟试路材料。",
        ],
    )


@router.get("/tasks/{task_id}", response_model=A02Task)
def get_trial_task(task_id: str) -> A02Task:
    return _get_task(task_id)


@router.post("/sessions", response_model=TrialSession)
def create_trial_session(request: TrialSessionCreateRequest) -> TrialSession:
    _get_task(request.task_id)
    return _trial_store().create_session(request.task_id)


@router.get("/sessions/{session_id}", response_model=TrialSession)
def get_trial_session(session_id: str) -> TrialSession:
    return _get_session(session_id)


@router.put("/sessions/{session_id}/answer", response_model=TrialSession)
def save_trial_answer(
    session_id: str,
    request: TrialAnswerUpdateRequest,
) -> TrialSession:
    try:
        return _trial_store().save_answer(session_id, request.answer)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="试路会话不存在。") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/sessions/{session_id}/event", response_model=TrialSession)
def reveal_trial_event(session_id: str) -> TrialSession:
    try:
        return _trial_store().reveal_event(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="试路会话不存在。") from exc


@router.post("/sessions/{session_id}/submit", response_model=TrialSession)
async def submit_trial_session(session_id: str) -> TrialSession:
    session = _get_session(session_id)
    if session.status == "submitted":
        return session
    _validate_answer(session.answer, event_revealed=session.event_revealed)
    try:
        evaluation = await _trial_agent().evaluate(A02_TASK, session.answer)
        observed_evidence = _observed_evidence(session)
        submitted = _trial_store().submit(session_id, observed_evidence, evaluation)
        _profile_store().record_observed_evidence(session_id, observed_evidence)
        return submitted
    except LLMGatewayError as exc:
        logger.warning("trial evaluation failed session_id=%s reason=%s", session_id, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        logger.warning("trial evaluation invalid session_id=%s reason=%s", session_id, exc)
        raise HTTPException(status_code=502, detail="Qwen 评价未通过结构化校验，请稍后重试。") from exc
