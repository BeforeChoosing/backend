"""Deterministic evidence coverage control for the profile exploration flow.

The exploration model can phrase a warm follow-up and extract explicit facts,
but it must not decide when an experience is ready for candidate cards.  This
module derives the seven coverage dimensions from user-authored text only.  It
is deliberately conservative: a lexical signal can make a dimension ``weak``
or ``sufficient`` for routing, but it never creates a fact or confirms a card.
"""

from dataclasses import dataclass
import re

from app.schemas.profile import (
    ExplorationCoverageStatus,
    ExplorationFocus,
    ProfileConversationMessage,
    ProfileExplorationRequest,
    ProfileExplorationResponse,
    StarDimension,
)


EXPLORATION_DIMENSIONS: tuple[ExplorationFocus, ...] = (
    "ownership",
    "decision",
    "constraint",
    "collaboration",
    "result",
    "transfer",
    "evidence",
)
CONTROLLER_VERSION = "profile-exploration-coverage-v2-star"

STAR_DIMENSIONS: tuple[StarDimension, ...] = ("S", "T", "A", "R")
_STOP_TERMS = ("不知道了", "停止", "结束", "不想继续", "直接总结", "不用再问")

_STATUS_RANK: dict[ExplorationCoverageStatus, int] = {
    "missing": 0,
    "weak": 1,
    "sufficient": 2,
    "confirmed": 3,
}

# Each dimension uses distinct lexical groups.  Matching one group is only a
# weak signal; two groups are sufficient for routing.  These are not a fact
# extractor and therefore never appear as evidence_found.
_CUE_GROUPS: dict[ExplorationFocus, tuple[tuple[str, ...], ...]] = {
    "ownership": (
        ("负责", "主导", "亲自", "本人", "我承担"),
        ("设计", "搭建", "实现", "推进", "整理", "访谈", "撰写", "跟进"),
        ("完成", "交付", "上线"),
    ),
    "decision": (
        ("决定", "选择", "取舍", "判断", "优先", "采用"),
        ("根据", "基于", "因为", "依据", "考虑到", "因此"),
        ("调整", "改为", "放弃", "保留", "方案"),
    ),
    "constraint": (
        ("限制", "约束", "预算", "资源", "时间", "周期", "截止"),
        ("困难", "风险", "冲突", "问题", "阻力", "不够", "只能", "无法"),
    ),
    "collaboration": (
        ("团队", "同事", "用户", "客户", "研发", "设计", "算法"),
        ("协作", "沟通", "协调", "推动", "分工", "一起", "共同"),
    ),
    "result": (
        ("结果", "最终", "影响", "反馈", "验证"),
        ("提升", "下降", "增长", "减少", "达到", "完成", "上线", "交付"),
        ("指标", "数据", "样本", "用户数", "转化", "留存", "准确率"),
    ),
    "transfer": (
        ("后来", "之后", "下一次", "以后", "如果再"),
        ("复用", "迁移", "类似", "应用到", "推广"),
        ("总结", "学到", "改进", "沉淀"),
    ),
    "evidence": (
        ("记录", "日志", "截图", "报告", "文档", "材料", "证据", "原型"),
        ("数据", "指标", "样本", "反馈", "问卷", "工单"),
    ),
}

_GAP_BY_FOCUS: dict[ExplorationFocus, str] = {
    "ownership": "补充你亲自负责的环节，以及你实际完成的动作。",
    "decision": "补充一次关键取舍，并说明你依据什么做出决定。",
    "constraint": "补充当时遇到的时间、资源或协作限制，以及你的处理方式。",
    "collaboration": "补充你与用户、同事或其他协作者如何分工并推动事情完成。",
    "result": "补充行动后的结果，优先提供可核对的变化、反馈或指标。",
    "transfer": "补充这次经历对下一次类似工作的具体影响。",
    "evidence": "补充能够核对的记录、数据、反馈或材料来源。",
}


@dataclass(frozen=True)
class ExplorationCoverage:
    """Coverage state derived from the current user-authored transcript."""

    status: dict[ExplorationFocus, ExplorationCoverageStatus]
    next_focus: ExplorationFocus
    ready_for_proposal: bool
    user_text_length: int
    star_dimension: StarDimension
    round_number: int
    next_action: str
    finalization_reason: str | None


def _contains_group(text: str, group: tuple[str, ...]) -> bool:
    return any(term in text for term in group)


def _user_authored_text(
    experience_text: str, messages: list[ProfileConversationMessage]
) -> str:
    # Assistant text is intentionally excluded.  It can contain invented
    # wording or instructions and must never increase evidence coverage.
    user_messages = [message.content for message in messages if message.role == "user"]
    return "\n".join([experience_text, *user_messages]).strip()


def _status_for_dimension(
    dimension: ExplorationFocus,
    text: str,
) -> ExplorationCoverageStatus:
    matched_groups = sum(_contains_group(text, group) for group in _CUE_GROUPS[dimension])
    if dimension == "evidence":
        # Numeric or explicit source signals are stronger than a generic word
        # such as “材料”, and make evidence sufficient with one cue group.
        numeric_signal = bool(
            re.search(r"(?:\d+(?:\.\d+)?\s*[%％]|\d+(?:\.\d+)?\s*(?:人|份|条|次|件|天|周|月|万|千))", text)
        )
        if numeric_signal:
            matched_groups += 1
    if matched_groups == 0:
        return "missing"
    if matched_groups == 1:
        return "weak"
    return "sufficient"


def assess_exploration(request: ProfileExplorationRequest) -> ExplorationCoverage:
    """Compute coverage and the next prompt dimension without a model call."""

    text = _user_authored_text(request.experience_text, request.messages)
    status = {
        dimension: _status_for_dimension(dimension, text)
        for dimension in EXPLORATION_DIMENSIONS
    }

    # Prefer uncovered core evidence, then context, and finally transfer.  A
    # dimension echoed in focus_history is skipped while another dimension is
    # still available, which prevents the same coaching target from repeating.
    history = set(request.focus_history)
    priority = {dimension: index for index, dimension in enumerate(EXPLORATION_DIMENSIONS)}
    candidates = [
        dimension
        for dimension in EXPLORATION_DIMENSIONS
        if _STATUS_RANK[status[dimension]] < _STATUS_RANK["sufficient"]
        and dimension not in history
    ]
    if not candidates:
        candidates = [
            dimension
            for dimension in EXPLORATION_DIMENSIONS
            if _STATUS_RANK[status[dimension]] < _STATUS_RANK["sufficient"]
        ]
    if not candidates:
        candidates = list(EXPLORATION_DIMENSIONS)
    next_focus = min(
        candidates,
        key=lambda dimension: (_STATUS_RANK[status[dimension]], priority[dimension]),
    )

    core_ready = all(
        _STATUS_RANK[status[dimension]] >= _STATUS_RANK["sufficient"]
        for dimension in ("ownership", "decision", "result", "evidence")
    )
    context_ready = any(
        _STATUS_RANK[status[dimension]] >= _STATUS_RANK["weak"]
        for dimension in ("constraint", "collaboration")
    )
    # The length guard keeps a single sentence with a few cue words from
    # unlocking card proposals.  It is deliberately below the API maximum so
    # a concise but concrete experience can pass after one or two turns.
    ready_for_proposal = len(text) >= 80 and core_ready and context_ready
    round_number = min(max(request.round_number, 1), 4)
    # Keep the four user-facing turns stable: S → T → A → R.  The model may
    # still phrase a more precise question inside that dimension, but it must
    # not skip a turn merely because the original experience happened to use a
    # cue word such as “目标” or “结果”.  A cue word is not proof that the
    # corresponding STAR evidence is complete.
    history = set(request.star_history)
    star_candidates = [dimension for dimension in STAR_DIMENSIONS if dimension not in history]
    star_dimension = star_candidates[0] if star_candidates else STAR_DIMENSIONS[-1]
    stop_requested = request.stop_requested or any(
        term in message.content for message in request.messages if message.role == "user" for term in _STOP_TERMS
    )
    next_action = "summarize" if stop_requested or round_number >= 4 else "ask"
    finalization_reason = (
        "用户选择停止补充" if stop_requested else "已完成四个 STAR 维度的追问" if round_number >= 4 else None
    )
    return ExplorationCoverage(
        status=status,
        next_focus=next_focus,
        ready_for_proposal=ready_for_proposal,
        user_text_length=len(text),
        star_dimension=star_dimension,
        round_number=round_number,
        next_action=next_action,
        finalization_reason=finalization_reason,
    )


def apply_exploration_controller(
    response: ProfileExplorationResponse,
    request: ProfileExplorationRequest,
) -> ProfileExplorationResponse:
    """Apply server-owned focus/readiness to a model response."""

    coverage = assess_exploration(request)
    if coverage.ready_for_proposal:
        gap = "核心证据已覆盖，可以整理候选能力卡；确认仍由你决定。"
    else:
        gap = _GAP_BY_FOCUS[coverage.next_focus]
    return response.model_copy(
        update={
            "focus_dimension": coverage.next_focus,
            "ready_for_proposal": coverage.ready_for_proposal,
            "coverage": coverage.status,
            "evidence_gap": gap,
            "star_dimension": coverage.star_dimension,
            "round_number": coverage.round_number,
            "next_action": coverage.next_action,
            "finalization_reason": coverage.finalization_reason,
        }
    )
