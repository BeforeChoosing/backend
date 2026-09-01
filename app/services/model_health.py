"""Small process-local circuit breaker for temporarily unhealthy models."""

from __future__ import annotations

from threading import Lock
from time import monotonic


class ModelHealthTracker:
    def __init__(self, *, failure_threshold: int = 2, cooldown_seconds: float = 60):
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_seconds = max(1.0, cooldown_seconds)
        self._lock = Lock()
        self._failures: dict[str, int] = {}
        self._disabled_until: dict[str, float] = {}

    def available(self, models: tuple[str, ...]) -> tuple[str, ...]:
        now = monotonic()
        with self._lock:
            available = tuple(
                model for model in models if self._disabled_until.get(model, 0) <= now
            )
            # Never deadlock a whole pool. If every circuit is open, allow the
            # scheduler to probe the pool again and refresh health from reality.
            return available or models

    def record_success(self, model: str) -> None:
        with self._lock:
            self._failures.pop(model, None)
            self._disabled_until.pop(model, None)

    def record_failure(self, model: str) -> bool:
        with self._lock:
            failures = self._failures.get(model, 0) + 1
            self._failures[model] = failures
            if failures < self.failure_threshold:
                return False
            self._disabled_until[model] = monotonic() + self.cooldown_seconds
            self._failures[model] = 0
            return True


_trackers: dict[tuple[int, float], ModelHealthTracker] = {}
_tracker_lock = Lock()


def get_model_health_tracker(
    *, failure_threshold: int = 2, cooldown_seconds: float = 60
) -> ModelHealthTracker:
    key = (max(1, failure_threshold), max(1.0, cooldown_seconds))
    with _tracker_lock:
        return _trackers.setdefault(
            key,
            ModelHealthTracker(
                failure_threshold=key[0],
                cooldown_seconds=key[1],
            ),
        )
