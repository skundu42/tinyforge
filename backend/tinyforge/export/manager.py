"""Export manager: fuse a LoRA adapter into its base, optionally convert to
MLX-quantized, and optionally push the result to the Hub.

For full from-scratch models (engine == "lm"), no fuse is needed — the run
output dir IS the complete HF model. For LoRA adapter runs (engine == "mlx"),
the existing fuse-then-optional-convert path is used.

Each export runs as a background job. The command runner, pusher, and run
resolver are injectable so the job lifecycle is testable without mlx or network.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tinyforge.export.commands import build_convert_command, build_fuse_command
from tinyforge.export.models import ExportRequest, ExportStatus

# run_command(cmd, cwd) -> (exit_code, combined_output)
RunCommandFn = Callable[[list[str], str], tuple[int, str]]
# run_resolver(run_id) -> (model_repo, adapter_path, engine)
RunResolver = Callable[[str], tuple[str, str, str]]
# push_fn(local_path, repo_id, base_model) -> hub url
PushFn = Callable[[str, str, str], str]


def _default_run_command(cmd: list[str], cwd: str) -> tuple[int, str]:
    import subprocess

    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return result.returncode, (result.stdout + result.stderr)


@dataclass
class _Export:
    id: str
    request: ExportRequest
    state: str = "running"
    error: str | None = None
    output_path: str | None = None
    hub_url: str | None = None


class ExportManager:
    def __init__(
        self,
        *,
        python_exe: str,
        exports_dir: Path,
        run_resolver: RunResolver,
        run_command: RunCommandFn = _default_run_command,
        push_fn: PushFn | None = None,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._python = python_exe
        self._exports_dir = Path(exports_dir)
        self._resolve_run = run_resolver
        self._run_command = run_command
        self._push_fn = push_fn
        self._id_factory = id_factory
        self._jobs: dict[str, _Export] = {}
        self._lock = threading.Lock()

    def start(self, request: ExportRequest) -> str:
        job = _Export(id=self._id_factory(), request=request)
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job.id

    def _run(self, job: _Export) -> None:
        import shutil

        request = job.request
        model_repo, adapter_path, engine = self._resolve_run(request.run_id)
        out_dir = self._exports_dir / job.id
        out_dir.mkdir(parents=True, exist_ok=True)

        if engine == "lm":
            # Full from-scratch model: the run dir IS the model — no fuse.
            if request.target == "safetensors":
                dest = out_dir / "model"
                try:
                    shutil.copytree(model_repo, dest, dirs_exist_ok=True)
                except OSError as exc:
                    return self._fail(job, f"copy failed: {exc}")
                result_path = str(dest)
            else:  # mlx
                mlx_path = out_dir / "mlx"
                code, output = self._run_command(
                    build_convert_command(self._python, model_repo, str(mlx_path), request.q_bits),
                    str(out_dir),
                )
                if code != 0:
                    return self._fail(job, output)
                result_path = str(mlx_path)
        else:
            # LoRA adapter on a base repo: fuse, then optionally convert.
            fused = out_dir / "fused"
            code, output = self._run_command(
                build_fuse_command(self._python, model_repo, adapter_path, str(fused)),
                str(out_dir),
            )
            if code != 0:
                return self._fail(job, output)
            result_path = str(fused)
            if request.target == "mlx":
                mlx_path = out_dir / "mlx"
                code, output = self._run_command(
                    build_convert_command(self._python, str(fused), str(mlx_path), request.q_bits),
                    str(out_dir),
                )
                if code != 0:
                    return self._fail(job, output)
                result_path = str(mlx_path)

        if request.push_repo:
            if self._push_fn is None:
                return self._fail(job, "push requested but no pusher configured")
            try:
                url = self._push_fn(result_path, request.push_repo, model_repo)
            except Exception as exc:  # noqa: BLE001
                return self._fail(job, f"push failed: {exc}")
            with self._lock:
                job.hub_url = url

        with self._lock:
            job.output_path = result_path
            job.state = "completed"

    def _fail(self, job: _Export, output: str) -> None:
        with self._lock:
            job.state = "failed"
            job.error = output[-800:] if output else "export failed"

    def status(self, job_id: str) -> ExportStatus:
        with self._lock:
            job = self._jobs[job_id]
            return ExportStatus(
                id=job.id, run_id=job.request.run_id, target=job.request.target,
                state=job.state, error=job.error, output_path=job.output_path,
                hub_url=job.hub_url,
            )

    def list(self) -> list[ExportStatus]:
        with self._lock:
            return [
                ExportStatus(
                    id=j.id, run_id=j.request.run_id, target=j.request.target, state=j.state,
                    error=j.error, output_path=j.output_path, hub_url=j.hub_url,
                )
                for j in self._jobs.values()
            ]
