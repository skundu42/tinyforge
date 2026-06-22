"""Tests for ExportManager (fuse/convert/push job, injectable command + push)."""

import time

from tinyforge.export.manager import ExportManager
from tinyforge.export.models import ExportRequest


def _wait(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def manager(tmp_path, *, run_command, push_fn=None, log=None):
    return ExportManager(
        python_exe="py", exports_dir=tmp_path,
        run_resolver=lambda run_id: ("base/m", f"/adapters/{run_id}"),
        run_command=run_command,
        push_fn=push_fn or (lambda path, repo, base: f"https://hf.co/{repo}"),
        id_factory=lambda: "exp1",
    )


def test_safetensors_export_runs_only_fuse(tmp_path) -> None:
    log: list[list[str]] = []
    mgr = manager(tmp_path, run_command=lambda cmd, cwd: (log.append(cmd) or (0, "")))

    job_id = mgr.start(ExportRequest(run_id="r1", target="safetensors"))
    assert _wait(lambda: mgr.status(job_id).state == "completed")

    assert len(log) == 1
    assert "fuse" in log[0]
    assert mgr.status(job_id).output_path.endswith("fused")


def test_mlx_export_runs_fuse_then_convert(tmp_path) -> None:
    log: list[list[str]] = []
    mgr = manager(tmp_path, run_command=lambda cmd, cwd: (log.append(cmd) or (0, "")))

    job_id = mgr.start(ExportRequest(run_id="r1", target="mlx", q_bits=4))
    assert _wait(lambda: mgr.status(job_id).state == "completed")

    assert len(log) == 2
    assert "convert" in log[1]
    assert mgr.status(job_id).output_path.endswith("mlx")


def test_gguf_export_passes_gguf_path(tmp_path) -> None:
    log: list[list[str]] = []
    mgr = manager(tmp_path, run_command=lambda cmd, cwd: (log.append(cmd) or (0, "")))

    job_id = mgr.start(ExportRequest(run_id="r1", target="gguf"))
    assert _wait(lambda: mgr.status(job_id).state == "completed")

    assert "--gguf-path" in log[0]


def test_fuse_failure_marks_failed_with_error(tmp_path) -> None:
    mgr = manager(tmp_path, run_command=lambda cmd, cwd: (1, "boom error detail"))

    job_id = mgr.start(ExportRequest(run_id="r1", target="safetensors"))
    assert _wait(lambda: mgr.status(job_id).state == "failed")
    assert "boom" in mgr.status(job_id).error


def test_push_invoked_when_repo_set(tmp_path) -> None:
    pushes: list[tuple] = []
    mgr = manager(
        tmp_path, run_command=lambda cmd, cwd: (0, ""),
        push_fn=lambda path, repo, base: (pushes.append((repo, base)) or "https://hf.co/me/m"),
    )

    job_id = mgr.start(ExportRequest(run_id="r1", target="safetensors", push_repo="me/m"))
    assert _wait(lambda: mgr.status(job_id).state == "completed")

    assert pushes == [("me/m", "base/m")]
    assert mgr.status(job_id).hub_url == "https://hf.co/me/m"
