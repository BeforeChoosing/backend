"""Generate candidate text-only TrialAgent answer cases through Bailian.

The default is intentionally explicit: this command is the only case
generation entry point and each task/quality-level pair costs at most one
cached Qwen request. ``--dry-run`` performs no network request and writes no
case that could be mistaken for a real answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.config import get_settings  # noqa: E402
from app.schemas.task_catalog import DynamicTrialAnswer  # noqa: E402
from app.services.llm_gateway import LLMGatewayError  # noqa: E402
from app.tasks.catalog import get_task_definition, list_task_definitions  # noqa: E402
from app.training.generation import CaseGenerator  # noqa: E402
from app.training.teacher import TeacherCache  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成固定任务库的文本作答案例候选")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/trial_agent/v1/case_inputs.local.jsonl"),
        help="案例输出 JSONL；默认写入本地忽略路径",
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="只生成指定任务，可重复传入；默认生成固定任务库全部任务",
    )
    parser.add_argument(
        "--levels",
        default="L1,L2,L3,L4,L5",
        help="质量级别，逗号分隔，默认 L1-L5",
    )
    parser.add_argument("--model", help="案例生成模型；默认读取 TRIAL_TEACHER_MODEL")
    parser.add_argument("--prompt-version", help="Prompt 版本；默认读取 TRIAL_TEACHER_PROMPT_VERSION")
    parser.add_argument("--cache", type=Path, help="SQLite 缓存路径；默认读取 TRIAL_TEACHER_CACHE_PATH")
    parser.add_argument("--limit", type=int, help="最多生成多少个任务/级别组合")
    parser.add_argument("--resume", action="store_true", help="跳过输出文件中已有的 case_id")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重新调用模型")
    parser.add_argument("--dry-run", action="store_true", help="仅检查参数并输出调用计划，不调用百炼")
    return parser.parse_args()


def _levels(value: str) -> list[str]:
    levels = [item.strip().upper() for item in value.split(",") if item.strip()]
    unknown = set(levels) - {"L1", "L2", "L3", "L4", "L5"}
    if unknown or not levels:
        raise ValueError("--levels 只能包含 L1、L2、L3、L4、L5")
    return list(dict.fromkeys(levels))


def _write(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    settings = get_settings()
    try:
        levels = _levels(args.levels)
        tasks = args.tasks or [task.id for task in list_task_definitions()]
        for task_id in tasks:
            get_task_definition(task_id)
    except (KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 1:
        print("--limit 必须大于 0", file=sys.stderr)
        return 2

    jobs = [(task_id, level) for task_id in tasks for level in levels]
    if args.resume and args.output.exists():
        existing_case_ids: set[str] = set()
        for line_number, line in enumerate(args.output.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"输出文件 {args.output} 第 {line_number} 行不是有效 JSON：{exc}", file=sys.stderr)
                return 2
            if isinstance(row, dict) and isinstance(row.get("case_id"), str):
                existing_case_ids.add(row["case_id"])
        jobs = [
            (task_id, level)
            for task_id, level in jobs
            if f"synthetic-{task_id.lower()}-{level.lower()}" not in existing_case_ids
        ]
    if args.limit is not None:
        jobs = jobs[: args.limit]
    model = (args.model or settings.trial_teacher_model).strip()
    prompt_version = (args.prompt_version or settings.trial_teacher_prompt_version).strip()
    cache_path = args.cache or Path(settings.trial_teacher_cache_path)
    print(f"任务/级别组合：{len(jobs)}")
    print(f"案例生成模型：{model}")
    print(f"Prompt 版本：{prompt_version}")
    print(f"缓存：{cache_path}")
    print(f"最多计划调用：{len(jobs)}（缓存命中时不调用）")
    if args.dry_run:
        for task_id, level in jobs:
            print(f"- {task_id} / {level}")
        return 0

    if not args.resume and args.output.exists():
        args.output.write_text("", encoding="utf-8")

    generator = CaseGenerator(
        settings=settings,
        cache=TeacherCache(cache_path),
    )
    failed = 0
    written = 0
    for index, (task_id, level) in enumerate(jobs, 1):
        try:
            response = generator.generate(
                task_id,
                quality_level=level,
                model=model,
                prompt_version=prompt_version,
                force=args.force,
            )
            if response.raw is None:
                raise ValueError("模型未返回答案 JSON")
            answer = DynamicTrialAnswer.model_validate(response.raw)
            case_id = f"synthetic-{task_id.lower()}-{level.lower()}"
            _write(
                args.output,
                {
                    "case_id": case_id,
                    "task_id": task_id,
                    "answer": answer.model_dump(mode="json"),
                    "confirmed_card_ids": [],
                    "metadata": {
                        "source": "qwen_case_generator",
                        "requested_quality_level": level,
                        "generator_model": response.model,
                        "generator_prompt_version": response.prompt_version,
                        "generator_request_fingerprint": response.fingerprint,
                        "cache_hit": response.cache_hit,
                        "multimodal_training": False,
                    },
                },
            )
            written += 1
            print(
                f"[{index}/{len(jobs)}] {case_id} -> 已写入 "
                f"({'缓存' if response.cache_hit else 'API'})"
            )
        except (KeyError, LLMGatewayError, ValueError, OSError) as exc:
            failed += 1
            print(f"[{index}/{len(jobs)}] {task_id}/{level} 失败：{exc}", file=sys.stderr)
    print(f"输出：{args.output}（{written} 条）")
    if failed:
        print(f"失败组合：{failed}，未写入伪造答案。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
