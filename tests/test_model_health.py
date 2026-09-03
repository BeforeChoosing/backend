from app.services.model_health import ModelHealthTracker


def test_model_health_opens_and_recovers_circuit() -> None:
    health = ModelHealthTracker(failure_threshold=2, cooldown_seconds=60)

    assert health.available(("model-a", "model-b")) == ("model-a", "model-b")
    assert health.record_failure("model-a") is False
    assert health.record_failure("model-a") is True
    assert health.available(("model-a", "model-b")) == ("model-b",)

    health.record_success("model-a")
    assert health.available(("model-a", "model-b")) == ("model-a", "model-b")


def test_model_health_never_deadlocks_an_entire_pool() -> None:
    health = ModelHealthTracker(failure_threshold=1, cooldown_seconds=60)
    health.record_failure("model-a")
    health.record_failure("model-b")

    assert health.available(("model-a", "model-b")) == ("model-a", "model-b")
