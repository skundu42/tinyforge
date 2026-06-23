"""Tests for InferenceService cancellation + generation serialization."""

import threading
import time

from tinyforge.infer.service import GenRequest, InferenceService


def req() -> GenRequest:
    return GenRequest(model_repo="m/x", prompt="hi")


def test_stream_stops_pulling_when_should_cancel_returns_true() -> None:
    produced: list[int] = []

    def gen(_request):
        for i in range(100):
            produced.append(i)
            yield str(i)

    service = InferenceService(generate_fn=gen)
    out: list[str] = []
    for token in service.stream(req(), should_cancel=lambda: len(out) >= 3):
        out.append(token)

    assert out == ["0", "1", "2"]
    # Stopped pulling from the underlying generator instead of running all 100.
    assert produced == [0, 1, 2, 3]


def test_stream_closes_underlying_generator_on_early_stop() -> None:
    closed = {"value": False}

    def gen(_request):
        try:
            for i in range(100):
                yield str(i)
        finally:
            closed["value"] = True  # real generate_fn releases the lock here

    service = InferenceService(generate_fn=gen)
    out: list[str] = []
    for token in service.stream(req(), should_cancel=lambda: len(out) >= 2):
        out.append(token)

    assert out == ["0", "1"]
    assert closed["value"] is True


def test_concurrent_streams_are_serialized() -> None:
    active = {"now": 0, "max": 0}
    guard = threading.Lock()

    def gen(_request):
        with guard:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        time.sleep(0.05)
        with guard:
            active["now"] -= 1
        yield "x"

    service = InferenceService(generate_fn=gen)
    threads = [threading.Thread(target=lambda: list(service.stream(req()))) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # The single-entry model cache + mlx are not thread-safe; generations must
    # not overlap.
    assert active["max"] == 1
