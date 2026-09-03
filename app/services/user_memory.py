"""Safely remove one account's persisted product memory."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from threading import Lock
from uuid import uuid4


_RESET_LOCK = Lock()
_MEMORY_TABLES = (
    "profile_cards",
    "profile_versions",
    "profile_evidence",
    "profile_conversation_turns",
    "profile_conversation_snapshots",
    "trial_sessions",
    "dynamic_trial_sessions",
    "profile_uploaded_materials",
    "model_response_cache",
)
_ACCOUNT_MEMORY_TABLES = (
    "profile_conversation_turns",
    "profile_conversation_snapshots",
)


def _record_count(db_path: Path) -> int:
    if not db_path.is_file():
        return 0
    with sqlite3.connect(str(db_path), timeout=10) as connection:
        existing = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        return sum(
            int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            for table in _MEMORY_TABLES
            if table in existing
        )


def _remove_tree(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _clear_account_memory(account_db_path: Path, user_id: str) -> int:
    if not account_db_path.is_file():
        return 0
    removed = 0
    with sqlite3.connect(str(account_db_path), timeout=10) as connection:
        existing = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        for table in _ACCOUNT_MEMORY_TABLES:
            if table not in existing:
                continue
            cursor = connection.execute(
                f'DELETE FROM "{table}" WHERE user_id = ?', (user_id,)
            )
            removed += max(0, cursor.rowcount)
    return removed


def reset_user_memory(
    db_path: str | Path,
    *,
    account_db_path: str | Path,
    user_id: str,
) -> tuple[int, int]:
    """Atomically detach and remove the current account's data and files."""

    target = Path(db_path).expanduser()
    files_dir = target.parent / f"{target.name}.files"
    with _RESET_LOCK:
        removed_records = _record_count(target) + _clear_account_memory(
            Path(account_db_path).expanduser(), user_id
        )
        removed_files = (
            sum(1 for item in files_dir.rglob("*") if item.is_file())
            if files_dir.is_dir()
            else 0
        )
        detached: list[Path] = []
        suffix = f".reset-{uuid4().hex}"
        for path in (files_dir, target, Path(f"{target}-wal"), Path(f"{target}-shm")):
            if not path.exists():
                continue
            tombstone = path.with_name(path.name + suffix)
            path.replace(tombstone)
            detached.append(tombstone)
        for path in detached:
            _remove_tree(path)
        return removed_records, removed_files
