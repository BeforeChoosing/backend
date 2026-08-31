"""Process-local FIFO admission queue for all DashScope chat requests.

The current deployment intentionally uses one Uvicorn worker.  A process-local
queue therefore gives the five-user demo deterministic ordering without adding
Redis.  If the API is later scaled to multiple workers, this module should be
backed by Redis while preserving the same public status contract.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Condition, Lock
from time import monotonic, time
from typing import Iterator


class LLMRequestCancelled(RuntimeError):
    """Raised when a user cancels a queued or running model request."""


@dataclass
class QueueTicket:
    request_id: str
    user_id: str
    state: str = "queued"
    enqueued_at: float = 0.0
    started_at: float | None = None
    cancel_requested: bool = False


class LLMRequestQueue:
    def __init__(self, *, max_concurrency: int, max_requests_per_minute: int):
        self.max_concurrency = max(1, max_concurrency)
        self.max_requests_per_minute = max(1, max_requests_per_minute)
        self._condition = Condition()
        self._waiting: deque[QueueTicket] = deque()
        self._active: dict[str, QueueTicket] = {}
        self._active_users: set[str] = set()
        self._starts: deque[float] = deque()

    def _trim_starts(self, now: float) -> None:
        while self._starts and now - self._starts[0] >= 60:
            self._starts.popleft()

    def _next_eligible(self) -> QueueTicket | None:
        for ticket in self._waiting:
            if ticket.user_id not in self._active_users:
                return ticket
        return None

    def _can_start(self, ticket: QueueTicket, now: float) -> bool:
        self._trim_starts(now)
        return (
            len(self._active) < self.max_concurrency
            and len(self._starts) < self.max_requests_per_minute
            and self._next_eligible() is ticket
        )

    @contextmanager
    def admission(self, *, request_id: str, user_id: str) -> Iterator[QueueTicket]:
        # Anonymous/background callers get request-scoped isolation rather than
        # sharing a single empty user key.
        effective_user = user_id or f"request:{request_id}"
        ticket = QueueTicket(
            request_id=request_id,
            user_id=effective_user,
            enqueued_at=time(),
        )
        with self._condition:
            self._waiting.append(ticket)
            self._condition.notify_all()
            while True:
                if ticket.cancel_requested:
                    self._remove_waiting(ticket)
                    ticket.state = "cancelled"
                    raise LLMRequestCancelled("请求已由用户取消。")
                now = monotonic()
                if self._can_start(ticket, now):
                    self._remove_waiting(ticket)
                    ticket.state = "running"
                    ticket.started_at = time()
                    self._active[ticket.request_id] = ticket
                    self._active_users.add(ticket.user_id)
                    self._starts.append(now)
                    break
                # Wake when an active request finishes or the RPM window moves.
                wait_seconds = 1.0
                if len(self._starts) >= self.max_requests_per_minute:
                    wait_seconds = min(1.0, max(0.05, 60 - (now - self._starts[0])))
                self._condition.wait(timeout=wait_seconds)

        try:
            yield ticket
            if ticket.cancel_requested:
                raise LLMRequestCancelled("请求已由用户取消。")
        finally:
            with self._condition:
                self._active.pop(ticket.request_id, None)
                self._active_users.discard(ticket.user_id)
                if ticket.state != "cancelled":
                    ticket.state = "completed"
                self._condition.notify_all()

    def _remove_waiting(self, ticket: QueueTicket) -> None:
        try:
            self._waiting.remove(ticket)
        except ValueError:
            pass

    def status_for_user(self, user_id: str) -> dict[str, object]:
        with self._condition:
            active = next(
                (ticket for ticket in self._active.values() if ticket.user_id == user_id),
                None,
            )
            if active is not None:
                return self._status(active, ahead=0)
            for index, ticket in enumerate(self._waiting):
                if ticket.user_id == user_id:
                    ahead = len(self._active) + index
                    return self._status(ticket, ahead=ahead)
            return {"state": "idle", "ahead": 0, "can_cancel": False}

    @staticmethod
    def _status(ticket: QueueTicket, *, ahead: int) -> dict[str, object]:
        return {
            "state": "cancelling" if ticket.cancel_requested else ticket.state,
            "ahead": max(0, ahead),
            "can_cancel": ticket.state in {"queued", "running"},
            "enqueued_at": ticket.enqueued_at,
            "started_at": ticket.started_at,
        }

    def cancel_for_user(self, user_id: str) -> bool:
        with self._condition:
            active = next(
                (ticket for ticket in self._active.values() if ticket.user_id == user_id),
                None,
            )
            if active is not None:
                active.cancel_requested = True
                self._condition.notify_all()
                return True
            for ticket in self._waiting:
                if ticket.user_id == user_id:
                    ticket.cancel_requested = True
                    self._remove_waiting(ticket)
                    ticket.state = "cancelled"
                    self._condition.notify_all()
                    return True
            return False


_queue: LLMRequestQueue | None = None
_queue_config: tuple[int, int] | None = None
_queue_lock = Lock()


def get_llm_request_queue(*, max_concurrency: int, max_requests_per_minute: int) -> LLMRequestQueue:
    global _queue, _queue_config
    config = (max(1, max_concurrency), max(1, max_requests_per_minute))
    with _queue_lock:
        if _queue is None or _queue_config != config:
            _queue = LLMRequestQueue(
                max_concurrency=config[0],
                max_requests_per_minute=config[1],
            )
            _queue_config = config
    return _queue
