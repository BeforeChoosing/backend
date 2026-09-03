"""Run the four P0 experiments and write a report suitable for the proposal."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.p0_experiments import (  # noqa: E402
    run_evidence_sensitivity_experiment,
    run_idempotency_experiment,
    run_rag_ablation_experiment,
    run_verifier_mutation_experiment,
)


def _percent(value: Any) -> str:
    return "—" if value is None else f"{float(value):.1%}"


def _render(report: dict[str, Any]) -> str:
    mutation = report["verifier_mutation"]
    rag = report["rag_ablation"]
    idempotency = report["idempotency"]
    lines = [
        "# P0 核心技术实验报告",
        "",
        f"生成时间：`{report['generated_at']}`",
        "",
        "## 1. 校验链变异测试",
        "",
        f"- 锁定案例：{mutation['case_count']} 条",
        f"- 自动注入异常：{mutation['attack_count']} 条",
        f"- 异常拦截率：{_percent(mutation['attack_detection_rate'])}",
        f"- 合法结果通过率：{_percent(mutation['valid_case_pass_rate'])}",
        f"- 误拒率：{_percent(mutation['false_rejection_rate'])}",
        "",
        "## 2. RAG 四组消融",
        "",
        "| 方案 | Hit@1 | Hit@3 | Hit@5 | MRR@5 | nDCG@5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "fts5": "FTS5",
        "vector": "纯向量",
        "fts_vector_fusion": "FTS5 + 向量融合",
        "fts_vector_rerank": "FTS5 + 向量融合 + 重排",
    }
    for arm, metrics in rag["arms"].items():
        lines.append(
            f"| {labels.get(arm, arm)} | {_percent(metrics['hit_at_1'])} | "
            f"{_percent(metrics['hit_at_3'])} | {_percent(metrics['hit_at_5'])} | "
            f"{_percent(metrics['mrr_at_5'])} | {_percent(metrics['ndcg_at_5'])} |"
        )
    lines += [
        "",
        f"- 查询数：{rag['query_count']} 条",
        f"- Embedding：`{rag['embedding_model']}`",
        f"- Rerank：`{rag['rerank_model']}`",
        f"- 新增 API 调用：{rag['api_calls']} 次",
        "",
        "## 3. 幂等与并发",
        "",
        f"- 并发相同请求：{idempotency['request_count']} 次",
        f"- 实际模型调用：{idempotency['actual_model_calls']} 次",
        f"- 避免重复调用：{idempotency['avoided_model_calls']} 次",
        f"- 调用减少率：{_percent(idempotency['call_reduction_rate'])}",
        f"- 返回结果一致率：{_percent(idempotency['result_consistency_rate'])}",
        "",
    ]
    sensitivity = report.get("evidence_sensitivity")
    if sensitivity:
        lines += [
            "## 4. TrialAgent 证据敏感性",
            "",
            f"- 原始/删证据配对：{sensitivity['pair_count']} 组",
            f"- 成功评价结果：{sensitivity['successful_result_count']} 条",
            f"- 唯一有效模型请求：{sensitivity['unique_successful_requests']} 次",
            f"- 当前运行新增调用 / 缓存命中：{sensitivity['api_calls_this_run']} / {sensitivity['cache_hits_this_run']} 次",
            f"- 原始作答平均分：{sensitivity['mean_original_score']:.2f}",
            f"- 删证据作答平均分：{sensitivity['mean_stripped_score']:.2f}",
            f"- 删证据后降分率：{_percent(sensitivity['score_drop_rate'])}",
            f"- 平均分数下降：{sensitivity['mean_score_delta']:.2f} 分",
            f"- 等级下降率：{_percent(sensitivity['level_drop_rate'])}",
            f"- 删证据后仍高分率：{_percent(sensitivity['unsupported_high_score_rate'])}",
            f"- 删证据结果触发二次校验率：{_percent(sensitivity['stripped_verifier_trigger_rate'])}",
            "",
            "| 任务 | 原始分 | 删证据分 | 下降 | 原始等级 | 删证据等级 |",
            "|---|---:|---:|---:|---|---|",
            *[
                f"| {row['task_id']} | {row['original']['weighted_score']:.2f} | "
                f"{row['evidence_stripped']['weighted_score']:.2f} | {row['score_delta']:.2f} | "
                f"{row['original']['observed_level']} | {row['evidence_stripped']['observed_level']} |"
                for row in sensitivity["pairs"]
            ],
            "",
        ]
    else:
        lines += [
            "## 4. TrialAgent 证据敏感性",
            "",
            "尚未执行。加入 `--live-sensitivity` 后运行 12 组配对、24 次 Qwen 调用。",
            "",
        ]
    lines += [
        "## 结论边界",
        "",
        "- 校验链、RAG 消融和幂等实验使用现有本地资产，不产生模型费用。",
        "- 证据敏感性实验只评价 12 个固定任务各一组配对，不代表所有自由输入。",
        "- 当前证据敏感性报告保留 24 个唯一有效模型结果；重复执行会命中本地缓存。",
        "- 幂等实验使用生产端点函数、真实指纹缓存和请求锁；模型替换为计数夹具，避免无意义付费。",
        "",
    ]
    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> int:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    mutation = run_verifier_mutation_experiment(args.cases)
    rag = run_rag_ablation_experiment(args.rag_cases, limit=args.rag_limit)
    idempotency = await run_idempotency_experiment(args.concurrent_requests)
    report: dict[str, Any] = {
        "report_version": "p0-experiments-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verifier_mutation": mutation,
        "rag_ablation": rag,
        "idempotency": idempotency,
    }
    if args.live_sensitivity:
        expected_calls = 24
        print(f"TrialAgent 证据敏感性预计最多调用 Qwen {expected_calls} 次；缓存命中不会重复调用。")
        report["evidence_sensitivity"] = await run_evidence_sensitivity_experiment(
            args.cases,
            output_dir=output / "evidence-sensitivity",
            concurrency=args.concurrency,
        )
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "report.md").write_text(_render(report), encoding="utf-8")
    print(f"已写入：{output / 'report.json'}")
    print(f"已写入：{output / 'report.md'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行四个 P0 技术实验")
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "datasets/trial_agent/eval/locked_cases.v1.jsonl",
    )
    parser.add_argument(
        "--rag-cases",
        type=Path,
        default=ROOT / "scripts/rag_eval_cases.json",
    )
    parser.add_argument("--rag-limit", type=int, default=5)
    parser.add_argument("--concurrent-requests", type=int, default=20)
    parser.add_argument("--live-sensitivity", action="store_true")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "evaluation-results/p0-experiments-v1",
    )
    return parser.parse_args()


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
