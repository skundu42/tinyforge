"""Tests for ExportManager: LoRA-adapter fuse/convert vs full-model (lm) export."""

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


def manager(tmp_path, *, run_command, engine="mlx", push_fn=None):
    # For lm runs, model_repo is the actual run dir under tmp_path so copytree works.
    # For mlx runs, model_repo is a symbolic "base/m" (not a real path, never copied).
    def _resolver(run_id):
        if engine == "lm":
            return str(tmp_path / "runs" / run_id), str(tmp_path / "runs" / run_id), engine
        return "base/m", f"/runs/{run_id}", engine

    return ExportManager(
        python_exe="py", exports_dir=tmp_path,
        run_resolver=_resolver,
        run_command=run_command,
        push_fn=push_fn or (lambda path, repo, base: f"https://hf.co/{repo}"),
        id_factory=lambda: "exp1",
    )


# --- LoRA-adapter (mlx) runs: existing fuse-based path -----------------------

def test_adapter_safetensors_runs_only_fuse(tmp_path) -> None:
    log: list[list[str]] = []
    mgr = manager(tmp_path, run_command=lambda cmd, cwd: (log.append(cmd) or (0, "")))
    job_id = mgr.start(ExportRequest(run_id="r1", target="safetensors"))
    assert _wait(lambda: mgr.status(job_id).state == "completed")
    assert len(log) == 1 and "fuse" in log[0]
    assert mgr.status(job_id).output_path.endswith("fused")


def test_adapter_mlx_runs_fuse_then_convert(tmp_path) -> None:
    log: list[list[str]] = []
    mgr = manager(tmp_path, run_command=lambda cmd, cwd: (log.append(cmd) or (0, "")))
    job_id = mgr.start(ExportRequest(run_id="r1", target="mlx", q_bits=4))
    assert _wait(lambda: mgr.status(job_id).state == "completed")
    assert len(log) == 2 and "convert" in log[1]
    assert mgr.status(job_id).output_path.endswith("mlx")


# --- Full-model (lm) runs: no fuse -------------------------------------------

def test_lm_safetensors_copies_run_dir_no_fuse(tmp_path) -> None:
    (tmp_path).joinpath("dummy").write_text("x")  # ensure tmp exists
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "model.safetensors").write_text("weights")
    log: list[list[str]] = []
    mgr = manager(tmp_path, run_command=lambda cmd, cwd: (log.append(cmd) or (0, "")), engine="lm")
    job_id = mgr.start(ExportRequest(run_id="r1", target="safetensors"))
    assert _wait(lambda: mgr.status(job_id).state == "completed")
    assert log == []  # no fuse, no subprocess
    out = mgr.status(job_id).output_path
    assert out.endswith("model")


def test_lm_mlx_converts_run_dir_directly(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    log: list[list[str]] = []
    mgr = manager(tmp_path, run_command=lambda cmd, cwd: (log.append(cmd) or (0, "")), engine="lm")
    job_id = mgr.start(ExportRequest(run_id="r1", target="mlx", q_bits=4))
    assert _wait(lambda: mgr.status(job_id).state == "completed")
    assert len(log) == 1 and "convert" in log[0]
    joined = " ".join(log[0])
    # Converts the run dir directly (not a fused dir) — path contains runs/r1
    assert "runs/r1" in joined and "--hf-path" in joined


def test_fuse_failure_marks_failed(tmp_path) -> None:
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
