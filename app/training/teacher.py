from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from app.config import Settings, get_settings
from app.services.llm_gateway import DashScopeQwenGateway
from app.training.cases import (
    TrialCaseInput,
    build_teacher_system_prompt,
    build_teacher_user_prompt,
    case_fingerprint,
)


_CACHE_LOCK_GUARD = Lock()
_CACHE_LOCKS: dict[tuple[str, str], Lock] = {}


@dataclass(frozen=True)
class TeacherResponse:
    raw: dict[str, Any] | None
    model: str
    prompt_version: str
    fingerprint: str
    cache_hit: bool
    api_calls: int
    status: str = "ok"
    error: str | None = None


class TeacherCache:
    """Small SQLite cache keyed by model, prompt version and case content."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS teacher_responses (
                    cache_key TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT response_json FROM teacher_responses WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        try:
            parsed = json.loads(row[0])
        except json.JSONDecodeError:
            # A partial local cache entry must not block a fresh, valid call.
            return None
        return parsed if isinstance(parsed, dict) else None

    @contextmanager
    def lock(self, cache_key: str):
        """Serialize same-key requests inside one local process."""

        lock_id = (str(self.path.resolve()), cache_key)
        with _CACHE_LOCK_GUARD:
            lock = _CACHE_LOCKS.setdefault(lock_id, Lock())
        with lock:
            yield

    def put(self, cache_key: str, *, model_id: str, prompt_version: str, response: dict[str, Any]) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT INTO teacher_responses(cache_key, model_id, prompt_version, response_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    model_id=excluded.model_id,
                    prompt_version=excluded.prompt_version,
                    response_json=excluded.response_json,
                    created_at=excluded.created_at
                """,
                (
                    cache_key,
                    model_id,
                    prompt_version,
                    json.dumps(response, ensure_ascii=False, sort_keys=True),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            connection.commit()

    def count(self) -> int:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute("SELECT COUNT(*) FROM teacher_responses").fetchone()
        return int(row[0]) if row else 0


def _cache_key(case: TrialCaseInput, *, model: str, prompt_version: str) -> str:
    return case_fingerprint(case, model=model, prompt_version=prompt_version)


class TeacherLabeler:
    """Call a text-only TrialAgent teacher with deterministic cache semantics."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        cache: TeacherCache | None = None,
        gateway: DashScopeQwenGateway | None = None,
    ):
        self.settings = settings or get_settings()
        self.cache = cache or TeacherCache(self.settings.trial_teacher_cache_path)
        self.gateway = gateway or DashScopeQwenGateway(self.settings)

    def label(
        self,
        case: TrialCaseInput,
        *,
        model: str | None = None,
        prompt_version: str | None = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> TeacherResponse:
        selected_model = (model or self.settings.trial_teacher_model).strip()
        selected_prompt_version = (prompt_version or self.settings.trial_teacher_prompt_version).strip()
        if not selected_model:
            raise ValueError("教师模型不能为空")
        fingerprint = _cache_key(case, model=selected_model, prompt_version=selected_prompt_version)
        with self.cache.lock(fingerprint):
            if not force:
                cached = self.cache.get(fingerprint)
                if cached is not None:
                    return TeacherResponse(
                        raw=cached,
                        model=selected_model,
                        prompt_version=selected_prompt_version,
                        fingerprint=fingerprint,
                        cache_hit=True,
                        api_calls=0,
                    )
            if dry_run:
                return TeacherResponse(
                    raw=None,
                    model=selected_model,
                    prompt_version=selected_prompt_version,
                    fingerprint=fingerprint,
                    cache_hit=False,
                    api_calls=0,
                    status="planned",
                )
            raw = self.gateway.generate_json(
                build_teacher_system_prompt(),
                build_teacher_user_prompt(case, prompt_version=selected_prompt_version),
                model=selected_model,
            )
            self.cache.put(
                fingerprint,
                model_id=selected_model,
                prompt_version=selected_prompt_version,
                response=raw,
            )
            return TeacherResponse(
                raw=raw,
                model=selected_model,
                prompt_version=selected_prompt_version,
                fingerprint=fingerprint,
                cache_hit=False,
                api_calls=1,
            )


def response_fingerprint(response: dict[str, Any]) -> str:
    payload = json.dumps(response, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
