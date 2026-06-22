"""Dataset splitting + a JSONL/SQLite registry of prepared datasets.

Prepared datasets are written as `train.jsonl` / `valid.jsonl` (the formats
mlx-lm reads) under a per-dataset directory; metadata lives in a SQLite table so
it survives restarts. The base dir is injectable for tests.
"""

from __future__ import annotations

import json
import random
import shutil
import sqlite3
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tinyforge.datasets.models import RegisteredDataset


def split_rows(
    rows: list[dict[str, Any]], val_fraction: float, seed: int = 0
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n_val = int(len(shuffled) * val_fraction)
    return shuffled[n_val:], shuffled[:n_val]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


class DatasetRegistry:
    def __init__(
        self,
        base_dir: Path,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
        clock: Callable[[], str] = _utc_now,
    ) -> None:
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self._id_factory = id_factory
        self._clock = clock
        self._db = self.base / "registry.db"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    target_format TEXT NOT NULL,
                    train_rows INTEGER NOT NULL,
                    val_rows INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    path TEXT NOT NULL
                )
                """
            )

    def save(
        self,
        name: str,
        rows: list[dict[str, Any]],
        target_format: str,
        val_fraction: float = 0.1,
        seed: int = 0,
    ) -> RegisteredDataset:
        train, val = split_rows(rows, val_fraction, seed)
        dataset_id = self._id_factory()
        dataset_dir = self.base / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(dataset_dir / "train.jsonl", train)
        _write_jsonl(dataset_dir / "valid.jsonl", val)

        record = RegisteredDataset(
            id=dataset_id, name=name, target_format=target_format,
            train_rows=len(train), val_rows=len(val), created_at=self._clock(),
            path=str(dataset_dir),
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO datasets VALUES (:id,:name,:target_format,:train_rows,:val_rows,:created_at,:path)",
                record.model_dump(),
            )
        return record

    def list(self) -> list[RegisteredDataset]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
        return [RegisteredDataset(**dict(row)) for row in rows]

    def get(self, dataset_id: str) -> RegisteredDataset:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        if row is None:
            raise KeyError(dataset_id)
        return RegisteredDataset(**dict(row))

    def delete(self, dataset_id: str) -> None:
        record = self.get(dataset_id)
        shutil.rmtree(record.path, ignore_errors=True)
        with self._connect() as conn:
            conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
