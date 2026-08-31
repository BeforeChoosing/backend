"""Resolve private storage from the authenticated server context, never a header.

The small deployment uses one SQLite file per account for profile snapshots,
task sessions/evidence and response caches. The original database remains the
account/audit database; pre-authentication legacy business rows stay untouched
and are intentionally not exposed or automatically adopted by a new account.
"""

from hashlib import sha256
from pathlib import Path

from fastapi import HTTPException

from app.services.request_context import RequestContext, get_request_context


def require_user_context() -> RequestContext:
    context = get_request_context()
    if not context.user_id:
        # Fail closed even if a new endpoint accidentally misses the middleware.
        raise HTTPException(status_code=401, detail="请先登录后再继续。")
    return context


def user_data_path(account_db_path: str | Path) -> Path:
    user_id = require_user_context().user_id
    base = Path(account_db_path).expanduser()
    # Fixed-length server-derived name prevents traversal and filename collisions.
    owner = sha256(user_id.encode("utf-8")).hexdigest()
    return base.parent / f"{base.name}.users" / f"{owner}.sqlite3"
