"""Load HuggingFace or local datasets and produce a preview.

`datasets` is injectable as `load_fn` so the mapping logic is testable without
network or files. Local files (json/csv/parquet) route through the matching
`datasets` builder; Hub datasets load by repo id.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tinyforge.datasets.models import DatasetPreview, DatasetSource


def _default_load(path: str, **kwargs: Any) -> Any:
    import datasets

    return datasets.load_dataset(path, **kwargs)


def load_dataset_for(source: DatasetSource, load_fn: Callable[..., Any] = _default_load) -> Any:
    if source.kind == "hub":
        if not source.repo_id:
            raise ValueError("hub source requires repo_id")
        return load_fn(source.repo_id, name=source.config, split=source.split)
    if not source.path:
        raise ValueError("local source requires path")
    fmt = source.file_format or _infer_format(source.path)
    return load_fn(fmt, data_files=source.path, split=source.split)


def preview(
    source: DatasetSource, limit: int = 20, load_fn: Callable[..., Any] = _default_load
) -> DatasetPreview:
    dataset = load_dataset_for(source, load_fn)
    num_rows = len(dataset)
    sample = dataset.select(range(min(limit, num_rows))).to_list()
    return DatasetPreview(columns=list(dataset.column_names), rows=sample, num_rows=num_rows)


def _infer_format(path: str) -> str:
    lowered = path.lower()
    if lowered.endswith(".csv"):
        return "csv"
    if lowered.endswith(".parquet"):
        return "parquet"
    return "json"  # .json / .jsonl
