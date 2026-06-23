"""Tests for TrainingService composition (runner + registry + dataset resolver)."""

from tinyforge.train.models import RunStatus, StartRunRequest
from tinyforge.train.registry import RunRegistry
from tinyforge.train.service import TrainingService


class FakeRunner:
    def __init__(self):
        self.started: dict = {}
        self._live = {}

    def start(self, config, run_id=None):
        self.started = {"config": config, "run_id": run_id}
        self._live[run_id] = "running"
        return run_id

    def status(self, run_id):
        if run_id not in self._live:
            raise KeyError(run_id)
        return RunStatus(id=run_id, name="t", state=self._live[run_id], num_events=3)

    def events(self, run_id, since=0):
        if run_id not in self._live:
            raise KeyError(run_id)
        return [{"event": "train", "iter": 1}][since:]

    def stop(self, run_id):
        self._live[run_id] = "stopped"

    def finish(self, run_id):
        self._live[run_id] = "completed"


def service(tmp_path):
    runner = FakeRunner()
    registry = RunRegistry(tmp_path / "runs")
    svc = TrainingService(
        runner=runner, registry=registry, runs_dir=tmp_path / "out",
        dataset_resolver=lambda ds_id: f"/data/{ds_id}",
        id_factory=lambda: "run1", clock=lambda: "t0",
    )
    return svc, runner, registry


def request(**kw) -> StartRunRequest:
    defaults = dict(name="exp", model_repo="m", dataset_id="ds1", iters=5)
    defaults.update(kw)
    return StartRunRequest(**defaults)


def test_start_resolves_dataset_and_persists_record(tmp_path) -> None:
    svc, runner, registry = service(tmp_path)

    record = svc.start(request())

    assert record.id == "run1"
    assert record.dataset_id == "ds1"
    assert runner.started["config"].data_dir == "/data/ds1"
    assert runner.started["config"].adapter_path.endswith("run1")
    assert runner.started["config"].iters == 5
    # persisted
    assert registry.get("run1").name == "exp"


def test_status_reflects_live_runner_and_syncs_registry(tmp_path) -> None:
    svc, runner, _ = service(tmp_path)
    svc.start(request())

    runner.finish("run1")
    status = svc.status("run1")

    assert status.state == "completed"
    # registry synced
    assert svc.get("run1").state == "completed"


def test_stop_marks_stopped(tmp_path) -> None:
    svc, runner, _ = service(tmp_path)
    svc.start(request())
    svc.stop("run1")
    assert svc.get("run1").state == "stopped"


def test_events_delegates_to_runner(tmp_path) -> None:
    svc, _, _ = service(tmp_path)
    svc.start(request())
    assert svc.events("run1") == [{"event": "train", "iter": 1}]


def test_lm_run_resolves_dataset_and_points_model_at_output_dir(tmp_path) -> None:
    svc, runner, registry = service(tmp_path)

    record = svc.start(request(engine="lm", model_size="tiny", dataset_id="dsX"))

    cfg = runner.started["config"]
    assert cfg.engine == "lm"
    assert cfg.data_dir == "/data/dsX"
    # from-scratch model lives in its own run dir, so model_repo == adapter_path
    assert cfg.model_repo == cfg.adapter_path
    # tiny preset applied
    assert (cfg.num_layers, cfg.hidden_size, cfg.num_heads, cfg.context_length) == (4, 128, 4, 256)
    assert record.engine == "lm"


def test_lm_custom_size_passes_knobs_through(tmp_path) -> None:
    svc, runner, _ = service(tmp_path)
    svc.start(request(
        engine="lm", model_size="custom", num_layers=3, hidden_size=192, num_heads=6,
        context_length=128,
    ))
    cfg = runner.started["config"]
    assert (cfg.num_layers, cfg.hidden_size, cfg.num_heads, cfg.context_length) == (3, 192, 6, 128)
