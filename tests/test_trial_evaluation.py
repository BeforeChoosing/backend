from datetime import datetime, timezone

from app.agents.trial_agent import TrialAgent
from app.schemas.task_catalog import DynamicTrialAnswer, DynamicTrialCoachUsage
from app.tasks.catalog import get_task_definition


def test_dynamic_evaluation_is_bound_to_source_rubric_and_coach_confidence() -> None:
    task = get_task_definition("M-02")
    answer = DynamicTrialAnswer(
        step_answers={step.id: "已完成" for step in task.steps},
        coach_usage=[
            DynamicTrialCoachUsage(
                level=3,
                prompt="半成品提示",
                used_at=datetime.now(timezone.utc),
            )
        ],
        event_decision="调整",
        event_response="提高高风险样本权重。",
    )
    raw = {
        "summary": "形成最小评测方案。",
        "dimensions": [
            {
                "dimension": criterion.dimension,
                "weight": 1,
                "score": 80,
                "evidence": "回答与该维度有关。",
            }
            for criterion in task.rubric
        ]
        + [{"dimension": "大模型自创维度", "score": 99, "evidence": "不应保留"}],
        "primary_ability": "错误能力",
        "observed_level": "L3",
        "level_reason": "覆盖正常、边界与高风险，并定义否决条件。",
        "supporting_evidence": [
            {"ability": task.supporting_skills[0], "observed_level": "L3", "evidence": "有证据"},
            {"ability": "大模型自创能力", "observed_level": "L5", "evidence": "不应保留"},
        ],
        "process_evidence": ["事件后调整"],
        "coach_dependency": "独立完成",
        "strengths": ["覆盖风险"],
        "gaps": ["样本一致性不足"],
        "next_step": "补评审一致性规则。",
        "confidence": "高",
    }

    evaluation = TrialAgent._normalize_dynamic(task, answer, raw)

    assert evaluation.primary_ability == "模型评测"
    assert [item.dimension for item in evaluation.dimensions] == [
        item.dimension for item in task.rubric
    ]
    assert [item.weight for item in evaluation.dimensions] == [
        item.weight for item in task.rubric
    ]
    assert evaluation.coach_dependency == "强提示"
    assert evaluation.confidence == "中"
    assert len(evaluation.supporting_evidence) == 1
