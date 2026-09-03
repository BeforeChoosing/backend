"""Local password accounts and bearer sessions for formal-mode access."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4


PASSWORD_ITERATIONS = 210_000
SESSION_TOKEN_BYTES = 32


class AuthStore:
    """Persist local accounts and revocable sessions in the profile database.

    This is intentionally a small local implementation for the on-device demo.
    Passwords are never stored in clear text and only a hash of each bearer token
    is persisted. The raw token is returned once to the browser after login.
    """

    def __init__(self, db_path: str | Path, *, session_ttl_hours: int = 24 * 30):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_ttl = timedelta(hours=max(1, int(session_ttl_hours)))
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
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS auth_users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS auth_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    revoked_at TEXT,
                    FOREIGN KEY(user_id) REFERENCES auth_users(id)
                );

                CREATE TABLE IF NOT EXISTS auth_password_reset_codes (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    code_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    consumed_at TEXT,
                    FOREIGN KEY(user_id) REFERENCES auth_users(id)
                );
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON auth_sessions(user_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_auth_sessions_expiry ON auth_sessions(expires_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_auth_reset_user ON auth_password_reset_codes(user_id, created_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_auth_reset_expiry ON auth_password_reset_codes(expires_at)"
            )

    @staticmethod
    def normalize_email(email: str) -> str:
        normalized = email.strip().lower()
        if "@" not in normalized or normalized.startswith("@") or normalized.endswith("@"):
            raise ValueError("请输入有效的邮箱地址。")
        local, domain = normalized.rsplit("@", 1)
        if not local or "." not in domain or domain.startswith(".") or domain.endswith("."):
            raise ValueError("请输入有效的邮箱地址。")
        return normalized

    @staticmethod
    def _encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")

    @staticmethod
    def _decode(value: str) -> bytes:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))

    @classmethod
    def _hash_password(cls, password: str) -> str:
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            PASSWORD_ITERATIONS,
        )
        return "$".join(
            (
                "pbkdf2_sha256",
                str(PASSWORD_ITERATIONS),
                cls._encode(salt),
                cls._encode(digest),
            )
        )

    @classmethod
    def _verify_password(cls, password: str, encoded: str) -> bool:
        try:
            algorithm, iterations, salt, expected = encoded.split("$", 3)
            if algorithm != "pbkdf2_sha256":
                return False
            digest = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                cls._decode(salt),
                int(iterations),
            )
            return hmac.compare_digest(cls._encode(digest), expected)
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _user_from_row(row: sqlite3.Row) -> dict[str, str]:
        return {
            "id": str(row["id"]),
            "email": str(row["email"]),
            "display_name": str(row["display_name"]),
        }

    def _create_session(self, connection: sqlite3.Connection, user_id: str) -> dict[str, object]:
        now = self._now()
        expires_at = now + self.session_ttl
        token = secrets.token_urlsafe(SESSION_TOKEN_BYTES)
        connection.execute(
            """
            INSERT INTO auth_sessions (token_hash, user_id, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (self._token_hash(token), user_id, now.isoformat(), expires_at.isoformat()),
        )
        return {"access_token": token, "expires_at": expires_at.isoformat()}

    def register(
        self,
        email: str,
        password: str,
        display_name: str | None = None,
    ) -> dict[str, object]:
        normalized_email = self.normalize_email(email)
        if len(password) < 8:
            raise ValueError("密码至少需要 8 个字符。")
        now = self._now()
        user_id = f"user-{uuid4().hex}"
        safe_name = (display_name or normalized_email.split("@", 1)[0]).strip()[:40]
        with self._connection() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO auth_users (
                        id, email, display_name, password_hash, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        normalized_email,
                        safe_name or "探索者",
                        self._hash_password(password),
                        now.isoformat(),
                        now.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("该邮箱已经注册，请直接登录。") from exc
            row = connection.execute(
                "SELECT id, email, display_name FROM auth_users WHERE id = ?", (user_id,)
            ).fetchone()
            assert row is not None
            session = self._create_session(connection, user_id)
            return {"user": self._user_from_row(row), **session}

    def login(self, email: str, password: str) -> dict[str, object]:
        normalized_email = self.normalize_email(email)
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT id, email, display_name, password_hash
                FROM auth_users WHERE email = ?
                """,
                (normalized_email,),
            ).fetchone()
            if row is None or not self._verify_password(password, str(row["password_hash"])):
                raise ValueError("邮箱或密码不正确。")
            session = self._create_session(connection, str(row["id"]))
            return {"user": self._user_from_row(row), **session}

    def create_password_reset_code(
        self,
        email: str,
        *,
        ttl_minutes: int = 10,
    ) -> tuple[str, str] | None:
        """Create a short-lived reset code without revealing account existence."""
        normalized_email = self.normalize_email(email)
        ttl = timedelta(minutes=max(1, int(ttl_minutes)))
        now = self._now()
        code = f"{secrets.randbelow(1_000_000):06d}"
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, email FROM auth_users WHERE email = ?",
                (normalized_email,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE auth_password_reset_codes
                SET consumed_at = COALESCE(consumed_at, ?)
                WHERE user_id = ? AND consumed_at IS NULL
                """,
                (now.isoformat(), str(row["id"])),
            )
            connection.execute(
                """
                INSERT INTO auth_password_reset_codes
                    (id, user_id, code_hash, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    f"reset-{uuid4().hex}",
                    str(row["id"]),
                    self._hash_password(code),
                    now.isoformat(),
                    (now + ttl).isoformat(),
                ),
            )
            return str(row["email"]), code

    def reset_password(
        self,
        email: str,
        code: str,
        new_password: str,
        *,
        max_attempts: int = 5,
    ) -> dict[str, object]:
        normalized_email = self.normalize_email(email)
        if len(new_password) < 8:
            raise ValueError("密码至少需要 8 个字符。")
        normalized_code = code.strip()
        if len(normalized_code) != 6 or not normalized_code.isdigit():
            raise ValueError("验证码不正确或已过期。")
        now = self._now()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT r.id AS reset_id, r.user_id, r.code_hash, r.expires_at, r.attempts,
                       u.id, u.email, u.display_name
                FROM auth_password_reset_codes AS r
                JOIN auth_users AS u ON u.id = r.user_id
                WHERE u.email = ? AND r.consumed_at IS NULL
                ORDER BY r.created_at DESC LIMIT 1
                """,
                (normalized_email,),
            ).fetchone()
            invalid = row is None
            if row is not None:
                try:
                    expired = datetime.fromisoformat(str(row["expires_at"])) <= now
                except ValueError:
                    expired = True
                invalid = (
                    expired
                    or int(row["attempts"]) >= max_attempts
                    or not self._verify_password(normalized_code, str(row["code_hash"]))
                )
                if invalid:
                    connection.execute(
                        "UPDATE auth_password_reset_codes SET attempts = attempts + 1 WHERE id = ?",
                        (str(row["reset_id"]),),
                    )
            if invalid:
                connection.commit()
                raise ValueError("验证码不正确或已过期。")
            user_id = str(row["user_id"])
            connection.execute(
                "UPDATE auth_users SET password_hash = ?, updated_at = ? WHERE id = ?",
                (self._hash_password(new_password), now.isoformat(), user_id),
            )
            connection.execute(
                "UPDATE auth_password_reset_codes SET consumed_at = ? WHERE id = ?",
                (now.isoformat(), str(row["reset_id"])),
            )
            connection.execute(
                "UPDATE auth_sessions SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL",
                (now.isoformat(), user_id),
            )
            return {"user": self._user_from_row(row)}

    def resolve_token(self, token: str | None) -> dict[str, str] | None:
        if not token or not token.strip():
            return None
        now = self._now()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT u.id, u.email, u.display_name, s.expires_at
                FROM auth_sessions AS s
                JOIN auth_users AS u ON u.id = s.user_id
                WHERE s.token_hash = ? AND s.revoked_at IS NULL
                """,
                (self._token_hash(token.strip()),),
            ).fetchone()
            if row is None:
                return None
            try:
                expires_at = datetime.fromisoformat(str(row["expires_at"]))
            except ValueError:
                return None
            if expires_at <= now:
                return None
            return self._user_from_row(row)

    def revoke_token(self, token: str | None) -> bool:
        if not token or not token.strip():
            return False
        with self._connection() as connection:
            updated = connection.execute(
                """
                UPDATE auth_sessions
                SET revoked_at = ?
                WHERE token_hash = ? AND revoked_at IS NULL
                """,
                (self._now().isoformat(), self._token_hash(token.strip())),
            )
            return updated.rowcount > 0
