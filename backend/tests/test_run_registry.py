"""Tests for the SQLite run registry."""

from tinyforge.train.models import RunRecord
from tinyforge.train.registry import RunRegistry


def record(run_id="r1", state="running") -> RunRecord:
    return RunRecord(
        id=run_id, name="t", model_repo="m", dataset_id="d", state=state,
        created_at="t", adapter_path="/p", config={"iters": 2},
    )


def test_save_and_get_roundtrips_config(tmp_path) -> None:
    registry = RunRegistry(tmp_path)
    registry.save(record())
    got = registry.get("r1")
    assert got.name == "t"
    assert got.config["iters"] == 2


def test_list_returns_all(tmp_path) -> None:
    registry = RunRegistry(tmp_path)
    registry.save(record(run_id="r1"))
    registry.save(record(run_id="r2"))
    assert {r.id for r in registry.list()} == {"r1", "r2"}


def test_update_state(tmp_path) -> None:
    registry = RunRegistry(tmp_path)
    registry.save(record())
    registry.update_state("r1", "completed", None)
    assert registry.get("r1").state == "completed"


def test_persists_across_instances(tmp_path) -> None:
    RunRegistry(tmp_path).save(record())
    assert RunRegistry(tmp_path).get("r1").state == "running"
