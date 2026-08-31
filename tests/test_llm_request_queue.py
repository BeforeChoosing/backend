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
