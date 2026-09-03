from __future__ import annotations

from threading import Event, Thread
from time import sleep

from app.services.llm_request_queue import LLMRequestCancelled, LLMRequestQueue


def test_queue_is_fifo_and_reports_number_ahead() -> None:
    queue = LLMRequestQueue(max_concurrency=1, max_requests_per_minute=20)
    first_started = Event()
    release_first = Event()
    second_started = Event()

    def first() -> None:
        with queue.admission(request_id="req-1", user_id="user-1"):
            first_started.set()
            release_first.wait(2)

    def second() -> None:
        with queue.admission(request_id="req-2", user_id="user-2"):
            second_started.set()

    thread_one = Thread(target=first)
    thread_two = Thread(target=second)
    thread_one.start()
    assert first_started.wait(1)
    thread_two.start()
    sleep(0.05)

    assert queue.status_for_user("user-1")["state"] == "running"
    queued_status = queue.status_for_user("user-2")
    assert queued_status["state"] == "queued"
    assert queued_status["ahead"] == 1
    assert queued_status["can_cancel"] is True
    assert not second_started.is_set()

    release_first.set()
    thread_one.join(1)
    thread_two.join(1)
    assert second_started.is_set()


def test_queued_request_can_be_cancelled_by_user() -> None:
    queue = LLMRequestQueue(max_concurrency=1, max_requests_per_minute=20)
    first_started = Event()
    release_first = Event()
    cancelled = Event()

    def first() -> None:
        with queue.admission(request_id="req-1", user_id="user-1"):
            first_started.set()
            release_first.wait(2)

    def second() -> None:
        try:
            with queue.admission(request_id="req-2", user_id="user-2"):
                raise AssertionError("cancelled ticket must never start")
        except LLMRequestCancelled:
            cancelled.set()

    thread_one = Thread(target=first)
    thread_two = Thread(target=second)
    thread_one.start()
    assert first_started.wait(1)
    thread_two.start()
    sleep(0.05)

    assert queue.cancel_for_user("user-2") is True
    assert cancelled.wait(1)
    assert queue.status_for_user("user-2") == {
        "state": "idle",
        "ahead": 0,
        "can_cancel": False,
    }

    release_first.set()
    thread_one.join(1)
    thread_two.join(1)


def test_same_user_requests_are_serialised_while_other_users_can_run() -> None:
    queue = LLMRequestQueue(max_concurrency=2, max_requests_per_minute=20)
    first_started = Event()
    release_first = Event()
    same_user_started = Event()
    other_user_started = Event()

    def run(request_id: str, user_id: str, started: Event, release: Event | None = None) -> None:
        with queue.admission(request_id=request_id, user_id=user_id):
            started.set()
            if release is not None:
                release.wait(2)

    first = Thread(target=run, args=("req-1", "user-1", first_started, release_first))
    same = Thread(target=run, args=("req-2", "user-1", same_user_started))
    other = Thread(target=run, args=("req-3", "user-2", other_user_started))
    first.start()
    assert first_started.wait(1)
    same.start()
    other.start()

    assert other_user_started.wait(1)
    assert not same_user_started.is_set()
    release_first.set()
    first.join(1)
    same.join(1)
    other.join(1)
    assert same_user_started.is_set()


def test_cancel_all_for_user_removes_queued_requests_and_marks_active() -> None:
    queue = LLMRequestQueue(max_concurrency=1, max_requests_per_minute=20)
    active_started = Event()
    release_active = Event()
    cancelled = [Event(), Event()]

    def active() -> None:
        try:
            with queue.admission(request_id="active", user_id="user-reset"):
                active_started.set()
                release_active.wait(2)
        except LLMRequestCancelled:
            cancelled[0].set()

    def waiting() -> None:
        try:
            with queue.admission(request_id="waiting", user_id="user-reset"):
                raise AssertionError("reset must cancel queued work")
        except LLMRequestCancelled:
            cancelled[1].set()

    active_thread = Thread(target=active)
    waiting_thread = Thread(target=waiting)
    active_thread.start()
    assert active_started.wait(1)
    waiting_thread.start()
    sleep(0.05)

    assert queue.cancel_all_for_user("user-reset") == 2
    assert cancelled[1].wait(1)
    release_active.set()
    assert cancelled[0].wait(1)
    active_thread.join(1)
    waiting_thread.join(1)
    assert queue.status_for_user("user-reset")["state"] == "idle"


def test_pool_routes_to_an_idle_model_before_queueing() -> None:
    queue = LLMRequestQueue(
        max_concurrency=2,
        max_requests_per_minute=20,
        model_max_concurrency=1,
    )
    first_started = Event()
    release_first = Event()
    selected: list[str | None] = []

    def first() -> None:
        with queue.admission(
            request_id="req-model-1",
            user_id="user-model-1",
            candidate_models=("model-a",),
            pool="text:fast",
        ) as ticket:
            selected.append(ticket.selected_model)
            first_started.set()
            release_first.wait(2)

    thread = Thread(target=first)
    thread.start()
    assert first_started.wait(1)
    with queue.admission(
        request_id="req-model-2",
        user_id="user-model-2",
        candidate_models=("model-a", "model-b"),
        pool="text:fast",
    ) as ticket:
        selected.append(ticket.selected_model)

    release_first.set()
    thread.join(1)
    assert selected == ["model-a", "model-b"]


def test_five_users_can_use_five_idle_models_concurrently() -> None:
    queue = LLMRequestQueue(
        max_concurrency=5,
        max_requests_per_minute=30,
        model_max_concurrency=1,
    )
    candidates = tuple(f"model-{index}" for index in range(5))
    started = [Event() for _ in range(5)]
    release = Event()
    selected: list[str | None] = []

    def run(index: int) -> None:
        with queue.admission(
            request_id=f"req-five-{index}",
            user_id=f"user-five-{index}",
            candidate_models=candidates,
            pool="text:balanced",
        ) as ticket:
            selected.append(ticket.selected_model)
            started[index].set()
            release.wait(2)

    threads = [Thread(target=run, args=(index,)) for index in range(5)]
    for thread in threads:
        thread.start()
    assert all(event.wait(1) for event in started)
    assert set(selected) == set(candidates)

    release.set()
    for thread in threads:
        thread.join(1)


def test_global_queue_caps_concurrent_starts_at_six() -> None:
    queue = LLMRequestQueue(
        max_concurrency=6,
        max_requests_per_minute=60,
        model_max_concurrency=1,
    )
    started = [Event() for _ in range(7)]
    release = Event()

    def run(index: int) -> None:
        with queue.admission(
            request_id=f"req-six-{index}",
            user_id=f"user-six-{index}",
            candidate_models=(f"model-six-{index}",),
            pool="text:shared",
        ):
            started[index].set()
            release.wait(2)

    threads = [Thread(target=run, args=(index,)) for index in range(7)]
    for thread in threads:
        thread.start()
    assert all(event.wait(1) for event in started[:6])
    sleep(0.05)
    assert not started[6].is_set()

    release.set()
    for thread in threads:
        thread.join(1)
    assert started[6].is_set()
