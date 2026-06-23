"""Tests for TrainingRunner event accumulation + status (fake subprocess)."""

import time

import tinyforge.train.runner as runner_mod
from tinyforge.train.models import RunConfig
from tinyforge.train.runner import TrainingRunner


class RecordingRegistry:
    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []

    def register(self, pid: int) -> None:
        self.events.append(("register", pid))

    def unregister(self, pid: int) -> None:
        self.events.append(("unregister", pid))


class FakeProc:
    def __init__(self, lines):
        self.stdout = iter(lines)
        self.terminated = False
        self._code = 0

    def terminate(self):
        self.terminated = True

    def wait(self):
        return self._code


def _wait(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def config(tmp_path, **kw) -> RunConfig:
    defaults = dict(
        name="t", model_repo="m", data_dir=str(tmp_path / "d"),
        adapter_path=str(tmp_path / "a"), iters=2,
    )
    defaults.update(kw)
    return RunConfig(**defaults)


def test_runner_accumulates_events_and_completes(tmp_path) -> None:
    lines = [
        "Starting training..., iters: 2",
        "Iter 1: Train loss 5.0, Learning Rate 1.0e-05, It/sec 1.0, Tokens/sec 10.0, Trained Tokens 5, Peak mem 0.1 GB",
        "Iter 2: Val loss 4.5, Val took 0.1s",
        "Saved final weights to x.safetensors.",
    ]
    runner = TrainingRunner(
        python_exe="py", spawn=lambda cmd, cwd: FakeProc(lines), id_factory=lambda: "run1"
    )

    run_id = runner.start(config(tmp_path))
    assert _wait(lambda: runner.status(run_id).state == "completed")

    events = runner.events(run_id)
    assert any(e["event"] == "train" for e in events)
    assert any(e["event"] == "val" for e in events)
    assert any(e["event"] == "saved" for e in events)
    assert (tmp_path / "a" / "events.jsonl").exists()


def test_runner_marks_failed_on_nonzero_exit(tmp_path) -> None:
    proc = FakeProc(["Loading datasets"])
    proc._code = 1
    runner = TrainingRunner(
        python_exe="py", spawn=lambda cmd, cwd: proc, id_factory=lambda: "run1"
    )

    run_id = runner.start(config(tmp_path))
    assert _wait(lambda: runner.status(run_id).state in ("failed", "completed"))
    assert runner.status(run_id).state == "failed"


def test_runner_registers_and_unregisters_child_pid(tmp_path) -> None:
    proc = FakeProc(
        ["Iter 1: Train loss 5.0, Learning Rate 1.0e-05, It/sec 1.0, "
         "Tokens/sec 10.0, Trained Tokens 5, Peak mem 0.1 GB"]
    )
    proc.pid = 4242
    registry = RecordingRegistry()
    runner = TrainingRunner(
        python_exe="py", spawn=lambda cmd, cwd: proc, id_factory=lambda: "run1",
        child_registry=registry,
    )

    run_id = runner.start(config(tmp_path))
    assert _wait(lambda: runner.status(run_id).state == "completed")

    # The child must be tracked while alive and dropped once it exits.
    assert registry.events == [("register", 4242), ("unregister", 4242)]


def test_default_spawn_starts_new_session(monkeypatch) -> None:
    captured: dict = {}

    class FakePopen:
        def __init__(self, command, **kwargs):
            captured.update(kwargs)
            self.pid = 1

    monkeypatch.setattr(runner_mod.subprocess, "Popen", FakePopen)
    runner_mod._default_spawn(["echo", "hi"], cwd="/tmp")

    # New session => child is its own process-group leader, so killing the group
    # reaps it and any workers it spawns.
    assert captured.get("start_new_session") is True


def test_events_since_returns_tail(tmp_path) -> None:
    lines = [
        "Iter 1: Train loss 5.0, Learning Rate 1.0e-05, It/sec 1.0, Tokens/sec 10.0, Trained Tokens 5, Peak mem 0.1 GB",
        "Iter 2: Train loss 4.0, Learning Rate 1.0e-05, It/sec 1.0, Tokens/sec 10.0, Trained Tokens 9, Peak mem 0.1 GB",
    ]
    runner = TrainingRunner(
        python_exe="py", spawn=lambda cmd, cwd: FakeProc(lines), id_factory=lambda: "run1"
    )
    run_id = runner.start(config(tmp_path))
    assert _wait(lambda: runner.status(run_id).state == "completed")

    assert len(runner.events(run_id, since=1)) == len(runner.events(run_id)) - 1
