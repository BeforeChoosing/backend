from fastapi.testclient import TestClient

from app.api import trial as trial_api
from app.main import app
from app.schemas.profile import CardProposal
from app.schemas.trial import (
    ReflectionChange,
    ReflectionProposal,
    TrialDimensionEvaluation,
    TrialEvaluation,
)
from app.services.dynamic_trial_store import DynamicTrialStore
from app.services.llm_gateway import LLMGatewayError
from app.services.profile_store import ProfileStore
from app.services.trial_store import TrialStore


class FakeTrialAgent:
    async def evaluate(self, task, answer):
        return TrialEvaluation(
            summary="完成了结构化归因与事件后重排。",
            dimensions=[
                TrialDimensionEvaluation(
                    dimension="AI 产品化",
                    score=80,
                    evidence="能区分多个系统层级并提出验证动作。",
                )
            ],
            strengths=["保留了不确定性"],
            gaps=["需要继续补充跨任务证据"],
            next_step="完成第二个不同 Task Atom 的试路任务。",
            confidence="中",
        )

    async def evaluate_dynamic(self, task, answer):
        return TrialEvaluation(
            summary="完成了固定五步交付与事件后调整。",
            dimensions=[
                TrialDimensionEvaluation(
                    dimension=item.dimension,
                    weight=item.weight,
                    score=82,
                    evidence="回答包含可追溯判断和事件后取舍。",
                )
                for item in task.rubric
            ],
            primary_ability=task.primary_skill,
            observed_level="L3",
            level_reason="能结合材料形成判断并给出最小验证。",
            process_evidence=["完成五步作答", "事件后调整了方案"],
            coach_dependency="方向性提示" if answer.coach_usage else "独立完成",
            strengths=["判断与验证对应"],
            gaps=["需要补充竞争性假设"],
            next_step="换一个 Task Atom 继续验证同一能力。",
            confidence="中",
        )


class FakeReflectionAgent:
    async def reflect(self, task, answer, evaluation, cards, previous_evidence):
        return ReflectionProposal(
            summary="本次任务形成了一条待继续验证的行为证据。",
            changes=[
                ReflectionChange(
                    change_type="新增证据",
                    ability=evaluation.primary_ability or task.primary_skill,
                    statement="用户完成了任务要求的判断和事件后调整。",
                    evidence_refs=["evaluation:level"],
                    basis=evaluation.level_reason or evaluation.summary,
                )
            ],
            next_verification=evaluation.next_step,
        )


class FailingReflectionAgent:
    async def reflect(self, task, answer, evaluation, cards, previous_evidence):
        raise LLMGatewayError("测试中的复盘模型故障")


def _complete_answer() -> dict:
    return {
        "attributions": [
            {"case_id": f"case-{index:02d}", "layer": "暂无法判断", "confidence": "中"}
            for index in range(1, 9)
        ],
        "priority_case_ids": ["case-01", "case-03"],
        "evidence": [
            {"source_id": "case-01", "source_type": "case", "explanation": "偏好跨轮丢失"},
            {"source_id": "case-03", "source_type": "case", "explanation": "未调用岗位库"},
        ],
        "validation_plans": [
            {"case_id": "case-01", "action": "检查状态读取日志", "expected_signal": "确认是否写入 Memory"},
            {"case_id": "case-03", "action": "检查 Tool 日志", "expected_signal": "区分策略与权限"},
        ],
        "event_decision": "调整",
        "event_priority_case_ids": ["case-03", "case-01"],
        "event_reason": "基础模型未退化后，优先验证工具和状态层。",
    }


def _confirm_trial_card(profile_store: ProfileStore) -> None:
    profile_store.confirm_cards([
        CardProposal(
            id="trial-card-1",
            title="拆解并验证产品方案",
            category="产品策略",
            description="根据约束缩小范围并设计验证动作",
            detail="来自用户已经确认的项目经历。",
            evidence_quote="根据约束缩小范围并设计验证动作",
            source_refs=["input:experience_text"],
            next_verification="在试路任务中继续验证",
            match_reason="与任务中的方案取舍相关",
            workplace_application="支持 AI 产品范围设计",
        )
    ])


def test_trial_api_requires_event_and_returns_observed_evidence(tmp_path, monkeypatch):
    trial_store = TrialStore(tmp_path / "trial.db")
    profile_store = ProfileStore(tmp_path / "profile.db")
    monkeypatch.setattr(trial_api, "_trial_store", lambda: trial_store)
    monkeypatch.setattr(trial_api, "_profile_store", lambda: profile_store)
    monkeypatch.setattr(trial_api, "_trial_agent", lambda: FakeTrialAgent())
    monkeypatch.setattr(trial_api, "_reflection_agent", lambda: FakeReflectionAgent())
    client = TestClient(app)

    preflight = client.options(
        "/api/v1/trial/sessions/test/answer",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "PUT",
        },
    )
    assert preflight.status_code == 200
    assert "PUT" in preflight.headers["access-control-allow-methods"]

    task = client.get("/api/v1/trial/tasks/A-02")
    assert task.status_code == 200
    assert len(task.json()["bad_cases"]) == 8

    session = client.post("/api/v1/trial/sessions", json={"task_id": "A-02"})
    assert session.status_code == 200
    session_id = session.json()["id"]

    saved = client.put(
        f"/api/v1/trial/sessions/{session_id}/answer",
        json={"answer": _complete_answer()},
    )
    assert saved.status_code == 200

    before_event = client.post(f"/api/v1/trial/sessions/{session_id}/submit")
    assert before_event.status_code == 422

    event = client.post(f"/api/v1/trial/sessions/{session_id}/event")
    assert event.status_code == 200
    submitted = client.post(f"/api/v1/trial/sessions/{session_id}/submit")
    assert submitted.status_code == 200
    payload = submitted.json()
    assert payload["status"] == "submitted"
    assert payload["observed_evidence"]["task_id"] == "A-02"
    assert payload["observed_evidence"]["reflection"]["profile_update_allowed"] is False
    assert profile_store.get_profile().version == 1


def test_dynamic_workbench_records_coach_and_qwen_evidence(tmp_path, monkeypatch):
    dynamic_store = DynamicTrialStore(tmp_path / "dynamic.db")
    profile_store = ProfileStore(tmp_path / "profile.db")
    _confirm_trial_card(profile_store)
    monkeypatch.setattr(trial_api, "_dynamic_trial_store", lambda: dynamic_store)
    monkeypatch.setattr(trial_api, "_profile_store", lambda: profile_store)
    monkeypatch.setattr(trial_api, "_trial_agent", lambda: FakeTrialAgent())
    monkeypatch.setattr(trial_api, "_reflection_agent", lambda: FailingReflectionAgent())
    client = TestClient(app)

    task = client.get("/api/v1/trial/catalog/F-01")
    assert task.status_code == 200
    assert sum(item["weight"] for item in task.json()["rubric"]) == 100
    assert set(task.json()["level_anchors"]) == {"L1", "L2", "L3", "L4", "L5"}

    created = client.post(
        "/api/v1/trial/workbench/sessions",
        json={"task_id": "F-01"},
    )
    session_id = created.json()["id"]
    coach = client.post(
        f"/api/v1/trial/workbench/sessions/{session_id}/coach",
        json={"level": 2},
    )
    assert coach.status_code == 200

    saved = client.put(
        f"/api/v1/trial/workbench/sessions/{session_id}/answer",
        json={
            "answer": {
                "selected_card_ids": ["trial-card-1"],
                "card_play_rationale": "用方案拆解能力划分首版范围。",
                "validation_hypothesis": "验证能否根据事件约束调整方案。",
                "card_play_completed": True,
                "step_answers": {
                    "problem": "优先解决素材整理阻塞。",
                    "evidence": "引用反馈与创作漏斗。",
                    "flow": "AI整理候选，用户确认；首版不自动发布。",
                    "validation": "测试完成率，低于基线则停止。",
                    "event": "周期缩短后收缩到一个整理节点。",
                },
                "viewed_material_ids": ["background", "constraints"],
                "evidence_refs": ["background", "constraints"],
                "event_decision": "调整",
                "event_response": "只保留一个模型调用节点。",
            }
        },
    )
    assert saved.status_code == 200
    assert len(saved.json()["answer"]["coach_usage"]) == 1

    client.post(f"/api/v1/trial/workbench/sessions/{session_id}/event")
    submitted = client.post(
        f"/api/v1/trial/workbench/sessions/{session_id}/submit"
    )
    assert submitted.status_code == 200
    evidence = submitted.json()["observed_evidence"]
    assert evidence["observed_level"] == "L3"
    assert evidence["primary_ability"] == "用户洞察"
    assert evidence["coach_dependency"] == "方向性提示"
    assert evidence["reflection"]["generation_mode"] == "deterministic_fallback"
    assert evidence["reflection"]["changes"][0]["change_type"] == "仍待验证"
    assert profile_store.get_completed_task_ids() == ["F-01"]


def test_dynamic_workbench_rejects_unconfirmed_card_play(tmp_path, monkeypatch):
    dynamic_store = DynamicTrialStore(tmp_path / "dynamic.db")
    profile_store = ProfileStore(tmp_path / "profile.db")
    monkeypatch.setattr(trial_api, "_dynamic_trial_store", lambda: dynamic_store)
    monkeypatch.setattr(trial_api, "_profile_store", lambda: profile_store)
    client = TestClient(app)

    created = client.post(
        "/api/v1/trial/workbench/sessions",
        json={"task_id": "F-01"},
    )
    session_id = created.json()["id"]
    saved = client.put(
        f"/api/v1/trial/workbench/sessions/{session_id}/answer",
        json={
            "answer": {
                "selected_card_ids": ["not-confirmed"],
                "card_play_rationale": "准备使用这张卡完成任务。",
                "validation_hypothesis": "验证能否完成任务中的关键判断。",
                "card_play_completed": True,
            }
        },
    )

    assert saved.status_code == 422
    assert saved.json()["detail"] == "能力出牌只能使用已确认的能力卡。"
