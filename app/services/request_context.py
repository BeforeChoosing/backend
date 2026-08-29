"""Request-scoped context shared by audit and model telemetry."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestContext:
    app_mode: str = "unknown"
    request_id: str = ""


_context: ContextVar[RequestContext] = ContextVar(
    "before_choosing_request_context",
    default=RequestContext(),
)


def set_request_context(context: RequestContext):
    return _context.set(context)


def reset_request_context(token) -> None:
    _context.reset(token)


def get_request_context() -> RequestContext:
    return _context.get()
