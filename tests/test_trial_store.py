from app.schemas.trial import A02Answer, A02Attribution, A02ValidationPlan
from app.services.trial_store import TrialStore


def _answer() -> A02Answer:
    return A02Answer(
        attributions=[
            A02Attribution(case_id=f"case-{index:02d}", layer="暂无法判断", confidence="中")
            for index in range(1, 9)
        ],
        priority_case_ids=["case-01", "case-03"],
        evidence=[
            {"source_id": "case-01", "source_type": "case", "explanation": "偏好跨轮丢失"},
            {"source_id": "case-03", "source_type": "case", "explanation": "未调用岗位库"},
        ],
        validation_plans=[
            {"case_id": "case-01", "action": "检查跨轮状态读取日志", "expected_signal": "确认偏好是否写入 Memory"},
            {"case_id": "case-03", "action": "检查 Tool 调用日志和权限", "expected_signal": "区分调用策略与权限失败"},
        ],
        event_decision="调整",
        event_priority_case_ids=["case-03", "case-01"],
        event_reason="模型未退化后，优先验证产品与工程层的工具和状态问题。",
    )


def test_trial_store_restores_answer_and_event(tmp_path):
    store = TrialStore(tmp_path / "profile.db")
    created = store.create_session("A-02")
    saved = store.save_answer(created.id, _answer())
    assert saved.answer.priority_case_ids == ["case-01", "case-03"]

    revealed = TrialStore(tmp_path / "profile.db").reveal_event(created.id)
    assert revealed.event_revealed is True
    assert TrialStore(tmp_path / "profile.db").get_session(created.id).answer.event_reason
