from fastapi.testclient import TestClient

from app.api import trial as trial_api
from app.main import app
from app.schemas.trial import TrialDimensionEvaluation, TrialEvaluation
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


def test_trial_api_requires_event_and_returns_observed_evidence(tmp_path, monkeypatch):
    trial_store = TrialStore(tmp_path / "trial.db")
    profile_store = ProfileStore(tmp_path / "profile.db")
    monkeypatch.setattr(trial_api, "_trial_store", lambda: trial_store)
    monkeypatch.setattr(trial_api, "_profile_store", lambda: profile_store)
    monkeypatch.setattr(trial_api, "_trial_agent", lambda: FakeTrialAgent())
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
    assert profile_store.get_profile().version == 1
