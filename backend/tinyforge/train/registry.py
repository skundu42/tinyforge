"""SQLite registry of training runs (metadata + state)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tinyforge.train.models import RunRecord


class RunRegistry:
    def __init__(self, base_dir: Path) -> None:
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self._db = self.base / "runs.db"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    model_repo TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    adapter_path TEXT NOT NULL,
                    config TEXT NOT NULL
                )
                """
            )

    def save(self, record: RunRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (:id,:name,:model_repo,:dataset_id,:state,:created_at,:adapter_path,:config)",
                {**record.model_dump(exclude={"config"}), "config": json.dumps(record.config)},
            )

    def update_state(self, run_id: str, state: str, error: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE runs SET state = ? WHERE id = ?", (state, run_id))

    def get(self, run_id: str) -> RunRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return self._to_record(row)

    def list(self) -> list[RunRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
        return [self._to_record(row) for row in rows]

    @staticmethod
    def _to_record(row: sqlite3.Row) -> RunRecord:
        data = dict(row)
        data["config"] = json.loads(data["config"])
        return RunRecord(**data)
