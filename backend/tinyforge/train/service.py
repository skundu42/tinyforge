"""TrainingService: orchestrates runs over the runner + registry.

Resolves a dataset id to its prepared data dir, assigns a run id + output
directory, starts the run, and persists metadata. Live state comes from the
runner (and is synced back to the registry); finished/old runs read from the
registry and their on-disk events.jsonl.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from tinyforge.train.models import RunConfig, RunRecord, RunStatus, StartRunRequest
from tinyforge.train.registry import RunRegistry
from tinyforge.train.runner import TrainingRunner


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrainingService:
    def __init__(
        self,
        *,
        runner: TrainingRunner,
        registry: RunRegistry,
        runs_dir: Path,
        dataset_resolver: Callable[[str], str],
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
        clock: Callable[[], str] = _utc_now,
    ) -> None:
        self._runner = runner
        self._registry = registry
        self._runs_dir = Path(runs_dir)
        self._resolve_dataset = dataset_resolver
        self._id_factory = id_factory
        self._clock = clock

    def start(self, request: StartRunRequest) -> RunRecord:
        run_id = self._id_factory()
        # The from-scratch torch engine needs no LLM model/dataset.
        data_dir = self._resolve_dataset(request.dataset_id) if request.engine == "mlx" else "(none)"
        model_repo = request.model_repo or ("(from-scratch MLP)" if request.engine == "torch" else "")
        adapter_path = str(self._runs_dir / run_id)
        config = RunConfig(
            name=request.name, model_repo=model_repo, data_dir=data_dir,
            adapter_path=adapter_path, engine=request.engine,
            fine_tune_type=request.fine_tune_type,
            num_layers=request.num_layers, batch_size=request.batch_size,
            iters=request.iters, learning_rate=request.learning_rate,
            steps_per_report=request.steps_per_report, steps_per_eval=request.steps_per_eval,
            max_seq_length=request.max_seq_length, grad_checkpoint=request.grad_checkpoint,
            seed=request.seed,
        )
        self._runner.start(config, run_id=run_id)
        record = RunRecord(
            id=run_id, name=request.name, model_repo=model_repo,
            dataset_id=request.dataset_id, state="running", created_at=self._clock(),
            adapter_path=adapter_path, config=config.model_dump(),
        )
        self._registry.save(record)
        return record

    def status(self, run_id: str) -> RunStatus:
        try:
            status = self._runner.status(run_id)
            self._registry.update_state(run_id, status.state, status.error)
            return status
        except KeyError:
            record = self._registry.get(run_id)
            return RunStatus(id=record.id, name=record.name, state=record.state)

    def events(self, run_id: str, since: int = 0) -> list[dict]:
        try:
            return self._runner.events(run_id, since)
        except KeyError:
            return self._read_events_file(run_id, since)

    def stop(self, run_id: str) -> None:
        self._runner.stop(run_id)
        self._registry.update_state(run_id, "stopped")

    def list(self) -> list[RunRecord]:
        records = self._registry.list()
        for record in records:
            try:
                status = self._runner.status(record.id)
                if status.state != record.state:
                    self._registry.update_state(record.id, status.state, status.error)
                    record.state = status.state
            except KeyError:
                pass
        return records

    def get(self, run_id: str) -> RunRecord:
        self.status(run_id)  # sync live state into the registry
        return self._registry.get(run_id)

    def _read_events_file(self, run_id: str, since: int) -> list[dict]:
        record = self._registry.get(run_id)
        path = Path(record.adapter_path) / "events.jsonl"
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as handle:
            events = [json.loads(line) for line in handle if line.strip()]
        return events[since:]
