from __future__ import annotations

from collections.abc import Sequence

from app.schemas.profile import ProfileCard
from app.schemas.task_catalog import DynamicTrialAnswer, TrialTaskDefinition
from app.schemas.trial import (
    TrialAbilityApplication,
    TrialAbilityEvidence,
    TrialDimensionEvaluation,
    TrialEvidenceBundle,
    TrialEvidenceItem,
    TrialEvaluation,
)


class TrialScoringService:
    """Build a server-owned evidence chain and constrain model scoring to it."""

    PROTOCOL_VERSION = "trial-evidence-v1"

    @classmethod
    def build_evidence(
        cls,
        task: TrialTaskDefinition,
        answer: DynamicTrialAnswer,
        cards: Sequence[ProfileCard],
    ) -> TrialEvidenceBundle:
        items: list[TrialEvidenceItem] = []
        seen_ids: set[str] = set()
        cards_by_id = {card.id: card for card in cards}
        selected_ids = list(dict.fromkeys(answer.selected_card_ids))
        challenges_by_id = {challenge.id: challenge for challenge in task.ability_challenges}
        materials_by_id = {material.id: material for material in task.materials}

        def add_item(
            item_id: str,
            source: str,
            source_id: str,
            kind: str,
            label: str,
            content: str,
        ) -> None:
            if item_id in seen_ids or not content.strip():
                return
            seen_ids.add(item_id)
            items.append(
                TrialEvidenceItem(
                    id=item_id,
                    source=source,  # type: ignore[arg-type]
                    source_id=source_id,
                    kind=kind,  # type: ignore[arg-type]
                    label=label,
                    content=content.strip()[:600],
                )
            )

        for card_id in selected_ids:
            card = cards_by_id.get(card_id)
            if card is None:
                continue
            add_item(
                f"card:{card.id}",
                "ability_card",
                card.id,
                "planned",
                "选择的能力卡",
                f"{card.title}：{card.description}",
            )

        for round_item in answer.card_play_rounds:
            challenge = challenges_by_id.get(round_item.challenge_id)
            if challenge is None:
                continue
            for card_id in round_item.selected_card_ids:
                card = cards_by_id.get(card_id)
                if card is None:
                    continue
                match_level = round_item.match_level or "未评价"
                add_item(
                    f"card_play:{round_item.challenge_id}:{card_id}",
                    "card_play",
                    round_item.challenge_id,
                    "planned",
                    f"{challenge.title} · 能力应用",
                    f"选择「{card.title}」，本轮匹配结果为{match_level}。{round_item.feedback}",
                )

        step_answer_refs: list[str] = []
        for step in task.steps:
            content = answer.step_answers.get(step.id, "").strip()
            if not content:
                continue
            item_id = f"answer:{step.id}"
            add_item(item_id, "answer", step.id, "deliverable", step.title, content)
            step_answer_refs.append(item_id)

        material_refs: list[str] = []
        for material_id in list(dict.fromkeys(answer.viewed_material_ids + answer.evidence_refs)):
            material = materials_by_id.get(material_id)
            if material is None:
                continue
            item_id = f"material:{material.id}"
            add_item(
                item_id,
                "material",
                material.id,
                "reference",
                material.title,
                material.content,
            )
            material_refs.append(item_id)

        if answer.card_play_rationale.strip():
            add_item(
                "answer:card_play_rationale",
                "answer",
                "card_play_rationale",
                "planned",
                "能力应用计划",
                answer.card_play_rationale,
            )
        if answer.validation_hypothesis.strip():
            add_item(
                "answer:validation_hypothesis",
                "answer",
                "validation_hypothesis",
                "planned",
                "待验证假设",
                answer.validation_hypothesis,
            )

        event_refs: list[str] = []
        if answer.event_decision is not None:
            add_item(
                "event:decision",
                "event",
                "event_decision",
                "observed",
                "事件后处理决定",
                answer.event_decision,
            )
            event_refs.append("event:decision")
        if answer.event_response.strip():
            add_item(
                "event:response",
                "event",
                "event_response",
                "deliverable",
                "事件后调整依据",
                answer.event_response,
            )
            event_refs.append("event:response")
        else:
            # The final workbench step now contains the decision and rationale
            # in one answer field. Keep that response in the event evidence
            # lane so the dynamic-adjustment rubric still sees it without
            # requiring a second decision form.
            final_step = task.steps[-1] if task.steps else None
            final_step_answer = (
                answer.step_answers.get(final_step.id, "").strip()
                if final_step is not None
                else ""
            )
            if final_step_answer:
                add_item(
                    "event:response",
                    "event",
                    final_step.id,
                    "deliverable",
                    "事件后重新决策",
                    final_step_answer,
                )
                event_refs.append("event:response")

        for index, usage in enumerate(answer.coach_usage, start=1):
            add_item(
                f"coach:{index}",
                "coach",
                f"level:{usage.level}",
                "interaction",
                f"Coach 提示 {index}",
                usage.prompt,
            )

        card_play_refs_by_card: dict[str, list[str]] = {card_id: [] for card_id in selected_ids}
        challenge_ids_by_card: dict[str, list[str]] = {card_id: [] for card_id in selected_ids}
        levels_by_card: dict[str, list[str]] = {card_id: [] for card_id in selected_ids}
        for round_item in answer.card_play_rounds:
            for card_id in round_item.selected_card_ids:
                if card_id not in card_play_refs_by_card:
                    continue
                card_play_refs_by_card[card_id].append(
                    f"card_play:{round_item.challenge_id}:{card_id}"
                )
                challenge_ids_by_card[card_id].append(round_item.challenge_id)
                levels_by_card[card_id].append(round_item.match_level or "未评价")

        ability_applications: list[TrialAbilityApplication] = []
        for card_id in selected_ids:
            card = cards_by_id.get(card_id)
            if card is None:
                continue
            card_play_refs = card_play_refs_by_card[card_id]
            challenge_ids = list(dict.fromkeys(challenge_ids_by_card[card_id]))
            levels = levels_by_card[card_id]
            if not card_play_refs:
                status = "未形成证据"
                basis = "能力卡已被选择，但三轮能力应用记录中没有使用它。"
            elif "high" in levels and len(step_answer_refs) >= 3:
                status = "已应用"
                basis = (
                    f"在{len(challenge_ids)}轮能力应用中得到直接匹配，且五步工作台已保存"
                    f"{len(step_answer_refs)}条作答。"
                )
            else:
                status = "部分应用"
                basis = (
                    f"在{len(challenge_ids)}轮能力应用中被选择，但当前交付物中仍缺少"
                    "足够的对应行为证据。"
                )
            application_refs = list(
                dict.fromkeys(
                    [f"card:{card_id}", *card_play_refs, *step_answer_refs[:3], *event_refs[:1]]
                )
            )[:8]
            next_step = {
                "已应用": "在下一项任务中记录该能力带来的具体结果，检验跨场景稳定性。",
                "部分应用": "在下一次作答中明确写出该能力如何影响判断和结果。",
                "未形成证据": "在任务交付物中补充该能力对应的具体行为和结果。",
            }[status]
            ability_applications.append(
                TrialAbilityApplication(
                    card_id=card.id,
                    card_title=card.title,
                    challenge_ids=challenge_ids,
                    evidence_refs=application_refs,
                    status=status,  # type: ignore[arg-type]
                    basis=basis,
                    next_step=next_step,
                )
            )

        card_refs = [f"card:{card_id}" for card_id in selected_ids if card_id in cards_by_id]
        card_play_refs = [item.id for item in items if item.source == "card_play"]
        summary_refs = list(
            dict.fromkeys(
                [*card_refs, *step_answer_refs, *event_refs, *material_refs[:1], *card_play_refs[:2]]
            )
        )[:12]
        return TrialEvidenceBundle(
            items=items,
            evidence_refs=summary_refs,
            selected_card_ids=[card_id for card_id in selected_ids if card_id in cards_by_id],
            ability_applications=ability_applications,
        )

    @classmethod
    def finalize_dynamic(
        cls,
        task: TrialTaskDefinition,
        answer: DynamicTrialAnswer,
        cards: Sequence[ProfileCard],
        evaluation: TrialEvaluation,
    ) -> tuple[TrialEvaluation, TrialEvidenceBundle]:
        bundle = cls.build_evidence(task, answer, cards)
        valid_refs = {item.id for item in bundle.items}
        answer_refs = [
            item.id
            for item in bundle.items
            if item.source == "answer" and item.kind == "deliverable"
        ]
        event_refs = [item.id for item in bundle.items if item.source == "event"]
        material_refs = [item.id for item in bundle.items if item.source == "material"]
        card_play_refs = [item.id for item in bundle.items if item.source == "card_play"]

        def defaults_for_dimension(dimension: str) -> list[str]:
            if dimension == "动态调整":
                return [*event_refs, *answer_refs[:2]]
            if dimension == "证据与推理":
                return [*material_refs[:3], *answer_refs[:2]]
            if dimension == task.primary_skill:
                return [*card_play_refs[:2], *answer_refs[:3]]
            return [*answer_refs[:3], *event_refs[:1], *material_refs[:1]]

        returned_dimensions = {item.dimension: item for item in evaluation.dimensions}
        dimensions: list[TrialDimensionEvaluation] = []
        for criterion in task.rubric:
            returned = returned_dimensions.get(criterion.dimension)
            refs = [ref for ref in (returned.evidence_refs if returned else []) if ref in valid_refs]
            refs = list(dict.fromkeys([*refs, *defaults_for_dimension(criterion.dimension)]))[:8]
            score = returned.score if returned else 0
            if not answer_refs:
                score = min(score, 35)
            elif not refs:
                score = min(score, 35)
            elif criterion.dimension == "动态调整" and not event_refs:
                score = min(score, 45)
            elif criterion.dimension == "证据与推理" and not material_refs:
                score = min(score, 70)
            evidence = (
                returned.evidence
                if returned and returned.evidence.strip()
                else f"关联 {len(refs)} 条可核对证据。"
            )
            dimensions.append(
                TrialDimensionEvaluation(
                    dimension=criterion.dimension,
                    weight=criterion.weight,
                    score=max(0, min(100, score)),
                    evidence=evidence,
                    evidence_refs=refs,
                )
            )

        supporting_evidence: list[TrialAbilityEvidence] = []
        allowed_supporting = set(task.supporting_skills[:2])
        for item in evaluation.supporting_evidence:
            if item.ability not in allowed_supporting:
                continue
            refs = [ref for ref in item.evidence_refs if ref in valid_refs]
            refs = list(dict.fromkeys([*refs, *answer_refs[:2]]))[:8]
            supporting_evidence.append(item.model_copy(update={"evidence_refs": refs}))
            if len(supporting_evidence) == 2:
                break

        observed_level = evaluation.observed_level
        if not answer_refs:
            observed_level = "证据不足"
        elif len(answer_refs) < 3 and observed_level in {"L3", "L4", "L5"}:
            observed_level = "L2"
        elif not event_refs and observed_level in {"L4", "L5"}:
            observed_level = "L3"

        finalized = evaluation.model_copy(
            update={
                "dimensions": dimensions,
                "supporting_evidence": supporting_evidence,
                "evidence_refs": bundle.evidence_refs,
                "ability_applications": bundle.ability_applications,
                "evaluation_protocol": cls.PROTOCOL_VERSION,
                "observed_level": observed_level,
            }
        )
        return finalized, bundle
