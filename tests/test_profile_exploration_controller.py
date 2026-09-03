from app.schemas.profile import ProfileConversationMessage, ProfileExplorationRequest
from app.services.profile_exploration_controller import (
    assess_exploration,
    apply_exploration_controller,
)
from app.schemas.profile import ProfileExplorationResponse


def _request(experience_text: str, *, messages=None, focus_history=None):
    return ProfileExplorationRequest(
        experience_text=experience_text,
        messages=messages or [],
        focus_history=focus_history or [],
    )


def test_controller_requires_core_evidence_before_proposal():
    assessment = assess_exploration(_request("我在项目中负责整理需求，完成了一版原型。"))

    assert assessment.ready_for_proposal is False
    assert assessment.status["ownership"] == "sufficient"
    assert assessment.status["decision"] == "missing"
    assert assessment.status["evidence"] in {"weak", "missing"}
    assert assessment.next_focus == "decision"


def test_controller_marks_rich_user_experience_ready():
    request = _request(
        "我在校园项目中主导用户访谈和原型搭建，面对两周周期和研发资源有限的约束，"
        "我根据15条反馈决定先解决首要问题，并与设计、研发协作完成上线。最终转化率提升20%，"
        "后来把这套访谈记录方法复用到另一个项目。"
    )

    assessment = assess_exploration(request)

    assert assessment.ready_for_proposal is True
    for dimension in ("ownership", "decision", "result", "evidence"):
        assert assessment.status[dimension] == "sufficient"


def test_controller_ignores_assistant_claims():
    request = _request(
        "我在项目中参与了日常工作，暂时没有整理具体结果。",
        messages=[
            ProfileConversationMessage(
                role="assistant",
                content="你主导了项目，基于数据做出取舍并带来20%的提升，已有完整证据。",
            )
        ],
    )

    assessment = assess_exploration(request)

    assert assessment.ready_for_proposal is False
    assert assessment.status["decision"] == "missing"
    assert assessment.status["evidence"] == "missing"


def test_controller_skips_dimension_echoed_in_focus_history():
    request = _request(
        "我在项目中负责整理需求并完成原型，团队协作推进上线。",
        focus_history=["decision"],
    )

    assessment = assess_exploration(request)

    assert assessment.next_focus != "decision"
    assert assessment.next_focus == "constraint"


def test_controller_overrides_model_readiness_and_focus():
    request = _request("我在校园项目中负责整理需求并完成一版可用原型。")
    response = ProfileExplorationResponse(
        trace_id="trace-controller",
        reply="补充经历中的具体依据。",
        focus_dimension="result",
        evidence_gap="模型认为材料已经足够。",
        ready_for_proposal=True,
    )

    controlled = apply_exploration_controller(response, request)

    assert controlled.ready_for_proposal is False
    assert controlled.focus_dimension == "decision"
    assert controlled.evidence_gap == "补充一次关键取舍，并说明你依据什么做出决定。"
    assert set(controlled.coverage) == {
        "ownership",
        "decision",
        "constraint",
        "collaboration",
        "result",
        "transfer",
        "evidence",
    }
