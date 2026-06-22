"""DatasetService: composes loading, formatting, tokenization, and the registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tinyforge.datasets.formatting import format_rows, target_format
from tinyforge.datasets.loader import _default_load, load_dataset_for, preview
from tinyforge.datasets.models import (
    DatasetPreview,
    DatasetSource,
    FormatSpec,
    RegisteredDataset,
    TokenStats,
)
from tinyforge.datasets.registry import DatasetRegistry
from tinyforge.datasets.tokenize import (
    compute_token_stats,
    make_length_fn,
    render_for_tokenization,
)


class DatasetService:
    def __init__(
        self,
        registry: DatasetRegistry,
        *,
        load_fn: Callable[..., Any] = _default_load,
        length_fn_factory: Callable[..., Callable[[str], int]] = make_length_fn,
    ) -> None:
        self._registry = registry
        self._load_fn = load_fn
        self._length_fn_factory = length_fn_factory

    def preview(self, source: DatasetSource, limit: int = 20) -> DatasetPreview:
        return preview(source, limit, self._load_fn)

    def analyze(
        self,
        source: DatasetSource,
        spec: FormatSpec,
        tokenizer_repo: str,
        sample: int = 200,
        token: str | None = None,
    ) -> TokenStats:
        rows = preview(source, sample, self._load_fn).rows
        target = target_format(spec)
        texts = [render_for_tokenization(r, target) for r in format_rows(rows, spec)]
        length_fn = self._length_fn_factory(tokenizer_repo, token)
        return compute_token_stats(texts, length_fn)

    def prepare(
        self,
        name: str,
        source: DatasetSource,
        spec: FormatSpec,
        val_fraction: float = 0.1,
        seed: int = 0,
        max_rows: int | None = None,
    ) -> RegisteredDataset:
        dataset = load_dataset_for(source, self._load_fn)
        count = len(dataset) if max_rows is None else min(len(dataset), max_rows)
        rows = dataset.select(range(count)).to_list()
        formatted = format_rows(rows, spec)
        return self._registry.save(name, formatted, target_format(spec), val_fraction, seed)

    def list(self) -> list[RegisteredDataset]:
        return self._registry.list()

    def get(self, dataset_id: str) -> RegisteredDataset:
        return self._registry.get(dataset_id)

    def delete(self, dataset_id: str) -> None:
        self._registry.delete(dataset_id)
