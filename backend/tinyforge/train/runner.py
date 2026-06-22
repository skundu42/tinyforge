"""Run mlx_lm.lora as an isolated subprocess, parsing output into events.

Each run is a child process so a native crash never takes down the orchestrator.
Output lines are parsed into structured events, appended in memory, and mirrored
to `<adapter_path>/events.jsonl`. `spawn` is injectable for testing.
"""

from __future__ import annotations

import json
import subprocess
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tinyforge.train.config import build_command
from tinyforge.train.models import RunConfig, RunStatus
from tinyforge.train.parser import parse_line


class ProcessLike:
    stdout: Any
    def terminate(self) -> None: ...  # pragma: no cover
    def wait(self) -> int: ...  # pragma: no cover


SpawnFn = Callable[[list[str], str], Any]


def _default_spawn(command: list[str], cwd: str) -> Any:
    return subprocess.Popen(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )


@dataclass
class _Run:
    id: str
    config: RunConfig
    state: str = "running"
    error: str | None = None
    events: list[dict] = field(default_factory=list)
    proc: Any = None


class TrainingRunner:
    def __init__(
        self,
        *,
        python_exe: str,
        spawn: SpawnFn = _default_spawn,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._python = python_exe
        self._spawn = spawn
        self._id_factory = id_factory
        self._runs: dict[str, _Run] = {}
        self._lock = threading.Lock()

    def start(self, config: RunConfig, run_id: str | None = None) -> str:
        Path(config.adapter_path).mkdir(parents=True, exist_ok=True)
        run = _Run(id=run_id or self._id_factory(), config=config)
        with self._lock:
            self._runs[run.id] = run
        command = build_command(config, self._python)
        threading.Thread(target=self._run, args=(run, command), daemon=True).start()
        return run.id

    def _run(self, run: _Run, command: list[str]) -> None:
        events_path = Path(run.config.adapter_path) / "events.jsonl"
        try:
            proc = self._spawn(command, run.config.adapter_path)
            run.proc = proc
            with open(events_path, "w", encoding="utf-8") as events_file:
                for line in proc.stdout:
                    event = parse_line(line)
                    if event is None:
                        continue
                    with self._lock:
                        run.events.append(event)
                    events_file.write(json.dumps(event) + "\n")
                    events_file.flush()
            code = proc.wait()
            with self._lock:
                if run.state != "stopped":
                    run.state = "completed" if code == 0 else "failed"
                    if code != 0 and run.error is None:
                        run.error = f"exited with code {code}"
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                run.state = "failed"
                run.error = str(exc)

    def stop(self, run_id: str) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.state = "stopped"
            proc = run.proc
        if proc is not None and hasattr(proc, "terminate"):
            try:
                proc.terminate()
            except (ProcessLookupError, OSError):
                pass

    def status(self, run_id: str) -> RunStatus:
        with self._lock:
            run = self._runs[run_id]
            return RunStatus(
                id=run.id, name=run.config.name, state=run.state,
                error=run.error, num_events=len(run.events),
            )

    def events(self, run_id: str, since: int = 0) -> list[dict]:
        with self._lock:
            return list(self._runs[run_id].events[since:])

    def config_of(self, run_id: str) -> RunConfig:
        with self._lock:
            return self._runs[run_id].config
