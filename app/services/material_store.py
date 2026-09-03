"""Private, per-account storage for user-uploaded source materials."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4


MAX_FILES_PER_USER = 50
MAX_BYTES_PER_USER = 200 * 1024 * 1024


class MaterialStorageError(ValueError):
    pass


class MaterialStore:
    def __init__(self, user_db_path: str | Path):
        self.db_path = Path(user_db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.files_dir = self.db_path.parent / f"{self.db_path.name}.files"
        self.files_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            self.files_dir.chmod(0o700)
        except OSError:
            pass
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.db_path), timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS profile_uploaded_materials (
                    id TEXT PRIMARY KEY,
                    original_name TEXT NOT NULL,
                    stored_name TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                )
                """
            )

    def save(self, *, original_name: str, data: bytes, mime_type: str) -> dict[str, object]:
        digest = hashlib.sha256(data).hexdigest()
        safe_original_name = Path(original_name).name[:240] or "upload"
        suffix = Path(safe_original_name).suffix.lower()
        if len(suffix) > 12 or not suffix.removeprefix(".").isalnum():
            suffix = ""
        with self._connection() as connection:
            existing = connection.execute(
                """
                SELECT id, original_name, stored_name, mime_type, size_bytes, sha256, created_at
                FROM profile_uploaded_materials WHERE sha256 = ?
                """,
                (digest,),
            ).fetchone()
            if existing is not None:
                return dict(existing)
            totals = connection.execute(
                """SELECT COUNT(*) AS count, COALESCE(SUM(size_bytes), 0) AS bytes
                   FROM profile_uploaded_materials"""
            ).fetchone()
            if int(totals["count"]) >= MAX_FILES_PER_USER:
                raise MaterialStorageError("已保存材料达到 50 份，请先整理已有材料。")
            if int(totals["bytes"]) + len(data) > MAX_BYTES_PER_USER:
                raise MaterialStorageError("账号材料总量不能超过 200MB。")

            material_id = f"material-{uuid4().hex}"
            stored_name = f"{material_id}{suffix}"
            target = self.files_dir / stored_name
            temporary = self.files_dir / f".{material_id}.tmp"
            temporary.write_bytes(data)
            try:
                temporary.chmod(0o600)
            except OSError:
                pass
            os.replace(temporary, target)
            created_at = datetime.now(timezone.utc).isoformat()
            try:
                connection.execute(
                    """
                    INSERT INTO profile_uploaded_materials (
                        id, original_name, stored_name, mime_type,
                        size_bytes, sha256, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        material_id,
                        safe_original_name,
                        stored_name,
                        mime_type or "application/octet-stream",
                        len(data),
                        digest,
                        created_at,
                    ),
                )
            except Exception:
                target.unlink(missing_ok=True)
                raise
        return {
            "id": material_id,
            "original_name": safe_original_name,
            "stored_name": stored_name,
            "mime_type": mime_type or "application/octet-stream",
            "size_bytes": len(data),
            "sha256": digest,
            "created_at": created_at,
        }

    def get(self, material_id: str) -> tuple[dict[str, object], Path] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT id, original_name, stored_name, mime_type, size_bytes, sha256, created_at
                FROM profile_uploaded_materials WHERE id = ?
                """,
                (material_id,),
            ).fetchone()
        if row is None:
            return None
        metadata = dict(row)
        path = self.files_dir / str(metadata["stored_name"])
        if not path.is_file():
            return None
        return metadata, path
