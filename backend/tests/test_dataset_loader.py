"""Tests for dataset loading + preview (datasets behind an injectable loader)."""

from datasets import Dataset

from tinyforge.datasets.loader import preview
from tinyforge.datasets.models import DatasetSource


def test_preview_returns_columns_rows_and_count() -> None:
    def fake_load(*args, **kwargs):
        return Dataset.from_dict({"text": ["a", "b", "c"], "label": [1, 2, 3]})

    result = preview(DatasetSource(kind="hub", repo_id="x"), limit=2, load_fn=fake_load)

    assert result.columns == ["text", "label"]
    assert result.num_rows == 3
    assert len(result.rows) == 2
    assert result.rows[0] == {"text": "a", "label": 1}


def test_preview_hub_source_passes_repo_and_split() -> None:
    captured: dict = {}

    def fake_load(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return Dataset.from_dict({"text": ["a"]})

    preview(
        DatasetSource(kind="hub", repo_id="org/ds", config="default", split="train"),
        load_fn=fake_load,
    )

    assert captured["path"] == "org/ds"
    assert captured["kwargs"]["name"] == "default"
    assert captured["kwargs"]["split"] == "train"


def test_preview_local_source_routes_through_format_loader() -> None:
    captured: dict = {}

    def fake_load(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return Dataset.from_dict({"text": ["a"]})

    preview(
        DatasetSource(kind="local", path="/tmp/data.jsonl", file_format="json"),
        load_fn=fake_load,
    )

    assert captured["path"] == "json"
    assert captured["kwargs"]["data_files"] == "/tmp/data.jsonl"
