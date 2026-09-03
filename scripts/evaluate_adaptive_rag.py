"""Evaluate the cost-aware adaptive RAG router.

The command keeps the knowledge base and vector index local.  Without
``--live`` it performs a zero-cost routing dry run using the existing cache;
with ``--live`` only low-margin queries can call the configured Bailian rerank
model, and every response is cached by the retriever.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.knowledge.hybrid import HybridKnowledgeRetriever  # noqa: E402
from app.knowledge.retriever import KnowledgeChunk  # noqa: E402


class _NoRemoteGateway:
    def embed(self, *args: object, **kwargs: object) -> object:
        raise RuntimeError("离线评测只允许读取已有 Embedding 缓存")

    def rerank(self, *args: object, **kwargs: object) -> object:
        raise RuntimeError("离线评测只允许读取已有 Rerank 缓存")


def _load_cases(path: Path) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("RAG 评测集必须是数组。")
    return [
        {"query": str(item["query"]), "expected_heading": str(item["expected_heading"])}
        for item in raw
        if isinstance(item, dict) and item.get("query") and item.get("expected_heading")
    ]


def _rank(results: list[KnowledgeChunk], expected: str) -> int | None:
    target = expected.lower()
    for index, chunk in enumerate(results, 1):
        if target in " > ".join(chunk.heading_path).lower():
            return index
    return None


def _metrics(ranks: list[int | None], limit: int) -> dict[str, float]:
    total = max(1, len(ranks))
    return {
        "hit_at_1": sum(rank == 1 for rank in ranks) / total,
        f"hit_at_{limit}": sum(rank is not None and rank <= limit for rank in ranks) / total,
        f"mrr_at_{limit}": sum(
            1 / rank for rank in ranks if rank is not None and rank <= limit
        )
        / total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="评估向量优先、低置信度重排的自适应 RAG")
    parser.add_argument("--cases", type=Path, default=ROOT / "scripts/rag_eval_cases.json")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--live", action="store_true", help="低置信度查询允许调用百炼 Rerank")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "evaluation-results/rag-adaptive-v1",
    )
    args = parser.parse_args()
    settings = get_settings()
    cases = _load_cases(args.cases)
    mode = "adaptive" if args.live else "adaptive"
    gateway = None if args.live else _NoRemoteGateway()
    retriever = HybridKnowledgeRetriever(
        settings.knowledge_dir,
        settings.knowledge_db_path,
        settings=replace(settings, rag_retriever_mode=mode),
        embedding_gateway=gateway,
        rerank_gateway=gateway,
    )
    ranks: list[int | None] = []
    details: list[dict[str, object]] = []
    rerank_calls = 0
    for case in cases:
        results = retriever.search(
            case["query"],
            corpus="career",
            document_id="job-ai-product-manager-v1",
            limit=args.limit,
        )
        rank = _rank(results, case["expected_heading"])
        ranks.append(rank)
        diagnostics = dict(retriever.last_diagnostics)
        rerank_calls += int(diagnostics.get("rerank_used", False))
        details.append(
            {
                "query": case["query"],
                "expected_heading": case["expected_heading"],
                "rank": rank,
                "mode": diagnostics.get("mode"),
                "adaptive_margin": diagnostics.get("adaptive_margin"),
                "rerank_used": diagnostics.get("rerank_used", False),
            }
        )
    report = {
        "report_version": "rag-adaptive-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query_count": len(cases),
        "k": args.limit,
        "retriever_mode": mode,
        "adaptive_margin": retriever.adaptive_margin,
        "adaptive_rerank_min_margin": retriever.adaptive_rerank_min_margin,
        "embedding_model": settings.bailian_embedding_model,
        "rerank_model": settings.bailian_rerank_model,
        "live": bool(args.live),
        "rerank_calls": rerank_calls,
        "metrics": _metrics(ranks, args.limit),
        "details": details,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = report["metrics"]
    (args.output_dir / "report.md").write_text(
        "# 自适应 RAG 评测\n\n"
        f"- 查询数：{len(cases)}\n"
        f"- 模式：`{mode}`\n"
        f"- Hit@1：{metrics['hit_at_1']:.1%}\n"
        f"- Hit@{args.limit}：{metrics[f'hit_at_{args.limit}']:.1%}\n"
        f"- MRR@{args.limit}：{metrics[f'mrr_at_{args.limit}']:.1%}\n"
        f"- 本次重排调用：{rerank_calls}\n"
        f"- 评测是否允许远端调用：{'是' if args.live else '否'}\n",
        encoding="utf-8",
    )
    print(json.dumps(report["metrics"], ensure_ascii=False))
    print(f"本次重排调用：{rerank_calls} 次；报告：{args.output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
