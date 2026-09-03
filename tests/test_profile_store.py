from app.schemas.profile import CardProposal, ProfileCardPatchRequest
from app.services.profile_store import ProfileStore


def _card(card_id: str = "card-1") -> CardProposal:
    return CardProposal(
        id=card_id,
        title="用户研究",
        category="洞察分析",
        description="通过访谈和反馈整理识别用户问题",
        detail="从用户反馈中提炼可行动问题。",
        icon="Eye",
        color_tone="purple",
        claim_level="interpretation",
        evidence_type="self_report",
        evidence_quote="访谈用户并根据反馈调整方案",
        source_refs=["input:experience_text"],
        pending_verification=True,
        next_verification="补充样本选择和决策变化",
        match_reason="依据用户自述的访谈和方案调整",
        workplace_application="支持 AI 功能需求分析",
    )


def test_profile_store_persists_versions_and_card_lifecycle(tmp_path):
    db_path = tmp_path / "profile.db"
    store = ProfileStore(db_path)

    confirmed = store.confirm_cards([_card()], trace_id="trace-1")
    assert confirmed.version == 1
    assert confirmed.cards[0].status == "confirmed"
    assert confirmed.cards[0].source_trace_id == "trace-1"

    reloaded = ProfileStore(db_path).get_profile()
    assert reloaded.version == 1
    assert reloaded.cards[0].title == "用户研究"

    updated = store.update_card(
        "card-1",
        ProfileCardPatchRequest(title="用户洞察", description="把反馈整理为可执行问题"),
    )
    assert updated.version == 2
    assert updated.cards[0].title == "用户洞察"
    assert updated.cards[0].description == "把反馈整理为可执行问题"

    deleted = store.delete_card("card-1")
    assert deleted.version == 3
    assert deleted.cards == []


def test_profile_store_rejects_missing_card(tmp_path):
    store = ProfileStore(tmp_path / "profile.db")

    try:
        store.update_card("missing", ProfileCardPatchRequest(title="不存在"))
    except KeyError as exc:
        assert exc.args == ("missing",)
    else:
        raise AssertionError("expected missing card to raise KeyError")


def test_profile_store_merges_new_experience_evidence_into_existing_card(tmp_path):
    store = ProfileStore(tmp_path / "profile.db")
    first = CardProposal.model_validate({
        **_card("card-existing").model_dump(),
        "title": "用户洞察能力",
        "experience_id": "experience-1",
        "source_refs": ["experience:experience-1"],
        "evidence_history": [{
            "experience_id": "experience-1",
            "evidence_quote": "访谈用户并根据反馈调整方案",
            "source_refs": ["experience:experience-1"],
        }],
    })
    store.confirm_cards([first], trace_id="trace-1")

    additional = CardProposal.model_validate({
        **_card("proposal-new").model_dump(),
        "title": "用户洞察能力",
        "experience_id": "experience-2",
        "evidence_quote": "整理客服反馈并调整需求优先级",
        "source_refs": ["experience:experience-2"],
        "resolution": "merge",
        "merge_target_card_id": "card-existing",
        "evidence_history": [{
            "experience_id": "experience-2",
            "evidence_quote": "整理客服反馈并调整需求优先级",
            "source_refs": ["experience:experience-2"],
        }],
    })
    response = store.confirm_cards([additional], trace_id="trace-2")

    assert len(response.cards) == 1
    assert response.cards[0].id == "card-existing"
    assert {item.experience_id for item in response.cards[0].evidence_history} == {
        "experience-1",
        "experience-2",
    }
    assert response.cards[0].source_refs == [
        "experience:experience-1",
        "experience:experience-2",
    ]
