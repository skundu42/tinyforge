"""Tests for dataset splitting + the JSONL/SQLite registry."""

import json

from tinyforge.datasets.registry import DatasetRegistry, split_rows


def test_split_rows_is_deterministic_with_seed() -> None:
    rows = [{"i": i} for i in range(10)]
    train, val = split_rows(rows, val_fraction=0.2, seed=0)
    assert len(train) == 8
    assert len(val) == 2

    train2, val2 = split_rows(rows, val_fraction=0.2, seed=0)
    assert train == train2
    assert val == val2


def test_save_writes_jsonl_splits_and_record(tmp_path) -> None:
    registry = DatasetRegistry(tmp_path, id_factory=lambda: "ds1", clock=lambda: "2026-01-01")
    rows = [{"text": f"r{i}"} for i in range(10)]

    record = registry.save("my-dataset", rows, "text", val_fraction=0.2, seed=0)

    assert record.id == "ds1"
    assert record.name == "my-dataset"
    assert record.target_format == "text"
    assert record.train_rows == 8
    assert record.val_rows == 2
    assert (tmp_path / "ds1" / "train.jsonl").exists()
    assert (tmp_path / "ds1" / "valid.jsonl").exists()

    train_lines = (tmp_path / "ds1" / "train.jsonl").read_text().strip().split("\n")
    assert len(train_lines) == 8
    assert "text" in json.loads(train_lines[0])


def test_list_and_get(tmp_path) -> None:
    registry = DatasetRegistry(tmp_path, id_factory=lambda: "ds1", clock=lambda: "t")
    registry.save("alpha", [{"text": "x"}], "text", val_fraction=0.0)

    items = registry.list()
    assert [d.name for d in items] == ["alpha"]
    assert registry.get("ds1").name == "alpha"


def test_delete_removes_files_and_record(tmp_path) -> None:
    registry = DatasetRegistry(tmp_path, id_factory=lambda: "ds1", clock=lambda: "t")
    registry.save("alpha", [{"text": "x"}], "text", val_fraction=0.0)

    registry.delete("ds1")

    assert registry.list() == []
    assert not (tmp_path / "ds1").exists()


def test_registry_persists_across_instances(tmp_path) -> None:
    DatasetRegistry(tmp_path, id_factory=lambda: "ds1", clock=lambda: "t").save(
        "alpha", [{"text": "x"}], "text", val_fraction=0.0
    )
    # A fresh registry over the same dir sees the persisted record.
    assert DatasetRegistry(tmp_path).get("ds1").name == "alpha"
