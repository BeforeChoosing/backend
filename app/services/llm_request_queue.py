"""Process-local FIFO admission queue for all DashScope chat requests.

The current deployment intentionally uses one Uvicorn worker.  A process-local
queue therefore gives the small demo deterministic ordering without adding
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
from random import SystemRandom
import logging

logger = logging.getLogger(__name__)


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
    requested_models: tuple[str, ...] = ()
    selected_model: str | None = None
    pool: str | None = None


class LLMRequestQueue:
    def __init__(
        self,
        *,
        max_concurrency: int,
        max_requests_per_minute: int,
        model_max_concurrency: int = 1,
    ):
        self.max_concurrency = max(1, max_concurrency)
        self.max_requests_per_minute = max(1, max_requests_per_minute)
        self.model_max_concurrency = max(1, model_max_concurrency)
        self._condition = Condition()
        self._waiting: deque[QueueTicket] = deque()
        self._active: dict[str, QueueTicket] = {}
        self._active_users: set[str] = set()
        self._starts: deque[float] = deque()
        self._model_starts: dict[str, deque[float]] = {}
        self._active_by_model: dict[str, int] = {}
        self._random = SystemRandom()

    def _trim_starts(self, now: float) -> None:
        while self._starts and now - self._starts[0] >= 60:
            self._starts.popleft()

    def _next_eligible(self) -> QueueTicket | None:
        for ticket in self._waiting:
            if ticket.user_id not in self._active_users:
                return ticket
        return None

    def _available_models(self, ticket: QueueTicket, now: float) -> list[str]:
        candidates = list(ticket.requested_models)
        if not candidates:
            return [""]
        available: list[str] = []
        for model in candidates:
            starts = self._model_starts.setdefault(model, deque())
            while starts and now - starts[0] >= 60:
                starts.popleft()
            if (
                self._active_by_model.get(model, 0) < self.model_max_concurrency
                and len(starts) < self.max_requests_per_minute
            ):
                available.append(model)
        return available

    def _can_start(self, ticket: QueueTicket, now: float) -> str | None:
        self._trim_starts(now)
        if (
            len(self._active) >= self.max_concurrency
            or len(self._starts) >= self.max_requests_per_minute
            or self._next_eligible() is not ticket
        ):
            return None
        available = self._available_models(ticket, now)
        if not available:
            return None
        return self._random.choice(available) if available[0] else ""

    @contextmanager
    def admission(
        self,
        *,
        request_id: str,
        user_id: str,
        candidate_models: tuple[str, ...] = (),
        pool: str | None = None,
    ) -> Iterator[QueueTicket]:
        # Anonymous/background callers get request-scoped isolation rather than
        # sharing a single empty user key.
        effective_user = user_id or f"request:{request_id}"
        ticket = QueueTicket(
            request_id=request_id,
            user_id=effective_user,
            enqueued_at=time(),
            requested_models=tuple(dict.fromkeys(candidate_models)),
            pool=pool,
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
                selected_model = self._can_start(ticket, now)
                if selected_model is not None:
                    self._remove_waiting(ticket)
                    ticket.state = "running"
                    ticket.started_at = time()
                    self._active[ticket.request_id] = ticket
                    self._active_users.add(ticket.user_id)
                    self._starts.append(now)
                    ticket.selected_model = selected_model or None
                    if selected_model:
                        self._active_by_model[selected_model] = (
                            self._active_by_model.get(selected_model, 0) + 1
                        )
                        self._model_starts.setdefault(selected_model, deque()).append(now)
                    logger.info(
                        "model request admitted request_id=%s pool=%s model=%s",
                        ticket.request_id,
                        ticket.pool or "shared",
                        ticket.selected_model or "unspecified",
                    )
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
                if ticket.selected_model:
                    active_count = self._active_by_model.get(ticket.selected_model, 0)
                    if active_count <= 1:
                        self._active_by_model.pop(ticket.selected_model, None)
                    else:
                        self._active_by_model[ticket.selected_model] = active_count - 1
                if ticket.state != "cancelled":
                    ticket.state = "completed"
                logger.info(
                    "model request released request_id=%s pool=%s model=%s state=%s",
                    ticket.request_id,
                    ticket.pool or "shared",
                    ticket.selected_model or "unspecified",
                    ticket.state,
                )
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
            "model": ticket.selected_model,
            "pool": ticket.pool,
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
_queue_config: tuple[int, int, int] | None = None
_queue_lock = Lock()


def get_llm_request_queue(
    *,
    max_concurrency: int,
    max_requests_per_minute: int,
    model_max_concurrency: int = 1,
) -> LLMRequestQueue:
    global _queue, _queue_config
    config = (
        max(1, max_concurrency),
        max(1, max_requests_per_minute),
        max(1, model_max_concurrency),
    )
    with _queue_lock:
        if _queue is None or _queue_config != config:
            _queue = LLMRequestQueue(
                max_concurrency=config[0],
                max_requests_per_minute=config[1],
                model_max_concurrency=config[2],
            )
            _queue_config = config
    return _queue
