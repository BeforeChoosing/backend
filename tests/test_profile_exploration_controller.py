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


def test_controller_advances_star_turns_in_order_and_summarizes_on_fourth():
    experience = "我在项目中负责访谈用户，后来根据反馈调整方案并完成上线。"
    dimensions = []
    for round_number in range(1, 5):
        request = ProfileExplorationRequest(
            experience_text=experience,
            round_number=round_number,
            star_history=dimensions,
        )
        assessment = assess_exploration(request)
        dimensions.append(assessment.star_dimension)
        assert assessment.star_dimension == ("S", "T", "A", "R")[round_number - 1]
        assert assessment.next_action == ("ask" if round_number < 4 else "summarize")


def test_controller_adds_non_fabricating_reply_direction_when_model_omits_it():
    request = ProfileExplorationRequest(
        experience_text="我参与过一个校园项目。",
        round_number=2,
        star_history=["S"],
    )
    response = ProfileExplorationResponse(
        trace_id="trace-suggestion-fallback",
        reply="接下来可以补充当时的目标。",
        focus_dimension="ownership",
        evidence_gap="仍缺少目标。",
    )

    controlled = apply_exploration_controller(response, request)

    assert controlled.star_dimension == "T"
    assert controlled.suggested_replies == [
        "我可以说明当时要完成的目标，以及我具体负责的部分。"
    ]


def test_controller_stop_intent_summarizes_without_answer_quality_gate():
    request = ProfileExplorationRequest(
        experience_text="我做过一个项目。",
        messages=[{"role": "user", "content": "不知道了"}],
    )

    assessment = assess_exploration(request)

    assert assessment.next_action == "summarize"
    assert assessment.finalization_reason == "用户选择停止补充"


def test_controller_keeps_four_round_boundary_in_supplement_only_mode():
    request = ProfileExplorationRequest(
        experience_text="我负责访谈用户并根据反馈调整方案，最终完成上线。",
        round_number=4,
        star_history=["S", "T", "A", "R"],
        supplement_only=True,
    )

    assessment = assess_exploration(request)

    assert assessment.next_action == "summarize"
    assert assessment.finalization_reason == "四轮后继续补充当前经历"
    assert assessment.round_number == 4
