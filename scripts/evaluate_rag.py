"""Compare local FTS5 retrieval with the Embedding + Rerank pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.config import get_settings  # noqa: E402
from app.knowledge.hybrid import HybridKnowledgeRetriever  # noqa: E402
from app.knowledge.retriever import KnowledgeChunk, KnowledgeRetriever  # noqa: E402
from app.knowledge.vector_index import LocalVectorIndex  # noqa: E402


@dataclass(frozen=True)
class EvalCase:
    query: str
    expected_heading: str


def _load_cases(path: Path) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("RAG 评测集必须是数组。")
    cases = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("query") or not item.get("expected_heading"):
            raise ValueError("RAG 评测集存在缺少 query 或 expected_heading 的条目。")
        cases.append(EvalCase(str(item["query"]), str(item["expected_heading"])))
    return cases


def _contains_expected(chunk: KnowledgeChunk, expected_heading: str) -> bool:
    return expected_heading.lower() in " > ".join(chunk.heading_path).lower()


def _evaluate(
    retriever: object,
    cases: Iterable[EvalCase],
    *,
    limit: int,
) -> tuple[float, float, list[str]]:
    hits = 0
    reciprocal_rank = 0.0
    details: list[str] = []
    for case in cases:
        results = retriever.search(
            case.query,
            corpus="career",
            document_id="job-ai-product-manager-v1",
            limit=limit,
        )
        rank = next(
            (
                index + 1
                for index, chunk in enumerate(results)
                if _contains_expected(chunk, case.expected_heading)
            ),
            None,
        )
        if rank is not None:
            hits += 1
            reciprocal_rank += 1 / rank
        details.append(f"{case.query} -> {rank or '未命中'}")
    total = max(1, len(details))
    return hits / total, reciprocal_rank / total, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比本地 FTS5 与 Embedding+Rerank 检索")
    parser.add_argument(
        "--live",
        action="store_true",
        help="调用一次向量查询和一次 Rerank/题目（使用已有本地向量索引）",
    )
    parser.add_argument("--limit", type=int, default=5, help="每个查询评估前 K 条结果")
    parser.add_argument(
        "--cases",
        type=Path,
        default=REPOSITORY_ROOT / "scripts" / "rag_eval_cases.json",
        help="评测集 JSON 路径",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    cases = _load_cases(args.cases)
    base = KnowledgeRetriever(settings.knowledge_dir, settings.knowledge_db_path)
    before_hit, before_mrr, before_details = _evaluate(base, cases, limit=args.limit)
    print(f"改造前 FTS5：Hit@{args.limit}={before_hit:.3f}，MRR@{args.limit}={before_mrr:.3f}")
    for detail in before_details:
        print(f"  {detail}")

    if not args.live:
        print("改造后：未执行（加入 --live 才调用百炼；默认不产生费用）")
        return 0

    index = LocalVectorIndex(settings.knowledge_db_path)
    if not index.ready:
        print("改造后无法评估：本地向量索引为空，请先运行 scripts/build_vector_index.py。", file=sys.stderr)
        return 1
    hybrid = HybridKnowledgeRetriever(
        settings.knowledge_dir,
        settings.knowledge_db_path,
        settings=settings,
    )
    after_hit, after_mrr, after_details = _evaluate(hybrid, cases, limit=args.limit)
    print(
        f"改造后 Embedding + {settings.bailian_rerank_model}："
        f"Hit@{args.limit}={after_hit:.3f}，MRR@{args.limit}={after_mrr:.3f}"
    )
    for detail in after_details:
        print(f"  {detail}")
    print(
        f"变化：Hit@{args.limit} {after_hit - before_hit:+.3f}，"
        f"MRR@{args.limit} {after_mrr - before_mrr:+.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
