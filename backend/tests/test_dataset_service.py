"""Tests for DatasetService composition (load → format → tokenize/register)."""

import json

from datasets import Dataset

from tinyforge.datasets.models import DatasetSource, FormatSpec
from tinyforge.datasets.registry import DatasetRegistry
from tinyforge.datasets.service import DatasetService


def fake_load(*args, **kwargs):
    return Dataset.from_dict(
        {"instruction": ["Add"], "input": ["1+1"], "output": ["2"], "text": ["hello world"]}
    )


def test_prepare_formats_rows_and_registers(tmp_path) -> None:
    registry = DatasetRegistry(tmp_path, id_factory=lambda: "ds1", clock=lambda: "t")
    service = DatasetService(registry, load_fn=fake_load)

    record = service.prepare(
        "math", DatasetSource(kind="hub", repo_id="x"), FormatSpec(mode="alpaca"),
        val_fraction=0.0,
    )

    assert record.target_format == "completion"
    assert record.train_rows == 1
    row = json.loads((tmp_path / "ds1" / "train.jsonl").read_text().strip())
    assert row["prompt"] == "Add\n\n1+1"
    assert row["completion"] == "2"


def test_analyze_computes_token_stats_with_injected_tokenizer(tmp_path) -> None:
    registry = DatasetRegistry(tmp_path)
    service = DatasetService(
        registry, load_fn=fake_load,
        length_fn_factory=lambda repo, token=None: (lambda text: len(text.split())),
    )

    stats = service.analyze(
        DatasetSource(kind="hub", repo_id="x"),
        FormatSpec(mode="text", text_column="text"),
        tokenizer_repo="x",
    )

    assert stats.count == 1
    assert stats.max == 2  # "hello world" -> 2 words
