"""Run local acceptance checks without calling Qwen unless explicitly requested."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.agents.career_agent import CareerAgent  # noqa: E402
from app.agents.profile_agent import ProfileAgent  # noqa: E402
from app.agents.reflection_agent import ReflectionAgent  # noqa: E402
from app.agents.trial_agent import TrialAgent  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402
from app.knowledge.hybrid import HybridKnowledgeRetriever  # noqa: E402
from app.knowledge.retriever import KnowledgeRetriever  # noqa: E402
from app.knowledge.vector_index import LocalVectorIndex  # noqa: E402
from app.services.llm_gateway import DashScopeQwenGateway  # noqa: E402
from app.services.bailian_retrieval import (  # noqa: E402
    DashScopeEmbeddingGateway,
    DashScopeRerankGateway,
)
from app.tasks.catalog import list_task_definitions  # noqa: E402


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def check_configuration(settings: Settings) -> CheckResult:
    missing = []
    if not settings.dashscope_api_key:
        missing.append("DASHSCOPE_API_KEY")
    if not settings.qwen_model.strip():
        missing.append("QWEN_MODEL")
    if not settings.bailian_embedding_model.strip():
        missing.append("BAILIAN_EMBEDDING_MODEL")
    if not settings.bailian_rerank_model.strip():
        missing.append("BAILIAN_RERANK_MODEL")

    parsed = urlparse(settings.dashscope_base_url)
    valid_url = (
        parsed.scheme == "https"
        and bool(parsed.netloc)
        and parsed.netloc.endswith("aliyuncs.com")
        and parsed.path.rstrip("/").endswith("/chat/completions")
    )
    if not valid_url:
        missing.append("DASHSCOPE_BASE_URL（需为百炼 OpenAI 兼容 chat/completions 地址）")

    if missing:
        return CheckResult("环境配置", False, "缺少或无效：" + "、".join(missing))
    return CheckResult(
        "环境配置",
        True,
        f"Qwen {settings.qwen_model}；Embedding {settings.bailian_embedding_model}；"
        f"Rerank {settings.bailian_rerank_model}；密钥已配置",
    )


def check_knowledge(settings: Settings) -> CheckResult:
    try:
        retriever = KnowledgeRetriever(settings.knowledge_dir, settings.knowledge_db_path)
        results = retriever.search_many(
            [
                "AI 产品经理 岗位职责 能力要求",
                "AI 产品经理 用户研究 产品方案",
            ],
            corpus="career",
            limit=3,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        return CheckResult("本地 RAG", False, str(exc))
    if retriever.chunk_count <= 0 or not results:
        return CheckResult("本地 RAG", False, "索引中没有可检索的岗位片段")
    vector_index = LocalVectorIndex(settings.knowledge_db_path)
    vector_detail = (
        f"本地向量 {vector_index.count} 条"
        if vector_index.ready
        else "本地向量尚未建立（可运行 scripts/build_vector_index.py）"
    )
    return CheckResult(
        "本地 RAG",
        True,
        f"FTS5 已索引 {retriever.chunk_count} 个片段，多意图检索返回 {len(results)} 条，"
        f"覆盖 {retriever.last_diagnostics.get('query_coverage', 0):.0%}；{vector_detail}",
    )


def check_task_catalog() -> CheckResult:
    tasks = list_task_definitions()
    valid = (
        len(tasks) == 12
        and len({task.id for task in tasks}) == 12
        and all(len(task.steps) == 5 for task in tasks)
        and all(sum(item.weight for item in task.rubric) == 100 for task in tasks)
    )
    if not valid:
        return CheckResult("试路任务库", False, "任务数量、五步结构或 Rubric 权重不完整")
    return CheckResult("试路任务库", True, "12 个固定任务均包含五步结构和完整 Rubric")


def check_agents(settings: Settings) -> CheckResult:
    gateway = DashScopeQwenGateway(settings)
    agents = [
        ProfileAgent(gateway),
        CareerAgent(gateway),
        TrialAgent(gateway),
        ReflectionAgent(gateway),
    ]
    if not all(getattr(agent, "PROMPT_VERSION", "") for agent in agents):
        return CheckResult("Agent", False, "存在未配置提示词版本的 Agent")
    names = "、".join(type(agent).__name__ for agent in agents)
    return CheckResult("Agent", True, names)


def check_http(name: str, url: str, *, expect_json_status: bool = False) -> CheckResult:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            body = response.read().decode("utf-8", errors="replace")
            status_code = response.status
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return CheckResult(name, False, f"无法连接 {url}：{exc}")
    if status_code != 200:
        return CheckResult(name, False, f"{url} 返回 HTTP {status_code}")
    if expect_json_status:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return CheckResult(name, False, "健康检查未返回 JSON")
        if payload.get("status") != "ok":
            return CheckResult(name, False, "健康检查状态不是 ok")
    return CheckResult(name, True, f"{url} 可访问")


def check_formal_auth_gate(health_url: str) -> CheckResult:
    """Verify that a formal business endpoint rejects anonymous access."""

    parsed = urlparse(health_url)
    path = parsed.path.rsplit("/", 1)[0] + "/profile/cards"
    url = parsed._replace(path=path, query="", fragment="").geturl()
    request = urllib.request.Request(url, headers={"X-App-Mode": "use"})
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            status_code = response.status
    except urllib.error.HTTPError as exc:
        status_code = exc.code
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return CheckResult("正式模式登录门禁", False, f"无法连接 {url}：{exc}")
    if status_code != 401:
        return CheckResult("正式模式登录门禁", False, f"匿名请求返回 HTTP {status_code}，应为 401")
    return CheckResult("正式模式登录门禁", True, "正式业务接口已拒绝未登录请求")


def check_live_qwen(settings: Settings) -> CheckResult:
    try:
        payload = DashScopeQwenGateway(settings).generate_json(
            "你是接口连通性检查助手。只输出 JSON 对象，不提供额外说明。",
            '{"task":"返回 status=ok 和 provider=Qwen"}',
        )
    except Exception as exc:  # noqa: BLE001 - surface gateway diagnostics in CLI
        return CheckResult("百炼真实调用", False, str(exc))
    if payload.get("status") != "ok":
        return CheckResult("百炼真实调用", False, "模型已响应，但未返回预期 status=ok")
    return CheckResult("百炼真实调用", True, "已完成 1 次 Qwen JSON 连通性调用")


def check_live_rag(settings: Settings) -> CheckResult:
    """Run one meaningful local-RAG query: one query embedding + one rerank."""
    try:
        index = LocalVectorIndex(settings.knowledge_db_path)
        if not index.ready:
            return CheckResult(
                "百炼 RAG 真实调用",
                False,
                "本地向量索引为空，请先运行 scripts/build_vector_index.py",
            )
        retriever = HybridKnowledgeRetriever(
            settings.knowledge_dir,
            settings.knowledge_db_path,
            settings=settings,
            embedding_gateway=DashScopeEmbeddingGateway(settings),
            rerank_gateway=DashScopeRerankGateway(settings),
        )
        results = retriever.search(
            "AI 产品经理如何验证用户需求并推动技术落地",
            corpus="career",
            document_id="job-ai-product-manager-v1",
            limit=3,
        )
    except Exception as exc:  # noqa: BLE001 - surface live diagnostics in CLI
        return CheckResult("百炼 RAG 真实调用", False, str(exc))
    if not results:
        return CheckResult("百炼 RAG 真实调用", False, "真实检索未返回岗位片段")
    return CheckResult(
        "百炼 RAG 真实调用",
        True,
        f"已完成 1 次 Embedding 查询和 1 次 {settings.bailian_rerank_model} 重排，"
        f"返回 {len(results)} 条（{retriever.last_diagnostics.get('mode', 'unknown')}）",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查本机 Demo 的配置与完整链路")
    parser.add_argument(
        "--backend-url",
        default="http://127.0.0.1:8000/api/v1/health",
        help="后端健康检查地址",
    )
    parser.add_argument(
        "--frontend-url",
        default="http://127.0.0.1:3000",
        help="前端页面地址",
    )
    parser.add_argument(
        "--skip-services",
        action="store_true",
        help="只检查配置、RAG、任务库和 Agent，不检查已启动服务",
    )
    parser.add_argument(
        "--live-qwen",
        action="store_true",
        help="额外执行 1 次会产生费用的 Qwen 连通性调用",
    )
    parser.add_argument(
        "--live-rag",
        action="store_true",
        help="额外执行 1 次 Embedding 查询和 1 次 Rerank，会产生少量费用",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    results = [
        check_configuration(settings),
        check_knowledge(settings),
        check_task_catalog(),
        check_agents(settings),
    ]
    if not args.skip_services:
        results.extend(
            [
                check_http("后端服务", args.backend_url, expect_json_status=True),
                check_formal_auth_gate(args.backend_url),
                check_http("前端服务", args.frontend_url),
            ]
        )
    if args.live_qwen:
        results.append(check_live_qwen(settings))
    if args.live_rag:
        results.append(check_live_rag(settings))

    for result in results:
        marker = "通过" if result.ok else "失败"
        print(f"[{marker}] {result.name}：{result.detail}")

    failures = [result for result in results if not result.ok]
    print(f"\n验收结果：{len(results) - len(failures)}/{len(results)} 项通过")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
