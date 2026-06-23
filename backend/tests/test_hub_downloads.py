"""Tests for the download manager (threaded download + poll-based progress)."""

import threading
import time

import pytest

from tinyforge.hub.downloads import DownloadManager


def _wait(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_plan_sums_will_download_and_cached_bytes() -> None:
    def plan_fn(repo_id, repo_type):
        return [
            ("a.safetensors", 1000, True),
            ("b.json", 50, True),
            ("c.bin", 999, False),  # already cached
        ]

    mgr = DownloadManager(plan_fn=plan_fn)
    plan = mgr.plan("meta/x")

    assert plan.total_bytes == 1050
    assert plan.already_cached_bytes == 999
    assert len(plan.files) == 3


def test_progress_accumulates_then_completes() -> None:
    gate = threading.Event()

    def download_fn(repo_id, repo_type, on_progress):
        on_progress(50)
        gate.wait(2.0)  # block so we can observe in-flight progress
        on_progress(50)
        return "/cache/meta/x"

    mgr = DownloadManager(
        plan_fn=lambda r, t: [("a", 100, True)],
        download_fn=download_fn,
        id_factory=lambda: "job1",
    )

    job_id = mgr.start("meta/x")

    assert _wait(lambda: mgr.progress(job_id).downloaded_bytes == 50)
    mid = mgr.progress(job_id)
    assert mid.state == "running"
    assert mid.fraction == 0.5

    gate.set()
    assert _wait(lambda: mgr.progress(job_id).state == "completed")
    done = mgr.progress(job_id)
    assert done.downloaded_bytes == 100
    assert done.fraction == 1.0
    assert done.local_path == "/cache/meta/x"


def test_start_records_error_on_failure() -> None:
    def download_fn(repo_id, repo_type, on_progress):
        raise RuntimeError("boom")

    mgr = DownloadManager(
        plan_fn=lambda r, t: [("a", 100, True)],
        download_fn=download_fn,
        id_factory=lambda: "job1",
    )

    job_id = mgr.start("meta/x")

    assert _wait(lambda: mgr.progress(job_id).state == "error")
    progress = mgr.progress(job_id)
    assert progress.error is not None and "boom" in progress.error


def test_progress_unknown_job_raises() -> None:
    mgr = DownloadManager(plan_fn=lambda r, t: [])
    with pytest.raises(KeyError):
        mgr.progress("nope")


class _FakeResponse:
    """Minimal stand-in for the requests.Response carried by HfHubHTTPError."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.headers: dict = {}
        self.request = None


def test_start_surfaces_friendly_message_when_model_is_gated() -> None:
    from huggingface_hub.errors import GatedRepoError

    def plan_fn(repo_id, repo_type):
        raise GatedRepoError(
            "401 Client Error. Cannot access gated repo", response=_FakeResponse(401)
        )

    mgr = DownloadManager(plan_fn=plan_fn, id_factory=lambda: "job1")
    # A gated repo must not crash the request handler with a bare 500: start()
    # records the failure as an error job instead of raising.
    job_id = mgr.start("meta-llama/Llama-3.2-1B-Instruct")

    progress = mgr.progress(job_id)
    assert progress.state == "error"
    assert progress.error is not None
    assert "gated" in progress.error.lower()
    assert "token" in progress.error.lower()


def test_start_surfaces_friendly_message_when_repo_missing() -> None:
    from huggingface_hub.errors import RepositoryNotFoundError

    def plan_fn(repo_id, repo_type):
        raise RepositoryNotFoundError("404 Client Error", response=_FakeResponse(404))

    mgr = DownloadManager(plan_fn=plan_fn, id_factory=lambda: "job1")
    job_id = mgr.start("nope/does-not-exist")

    progress = mgr.progress(job_id)
    assert progress.state == "error"
    assert progress.error is not None
    assert "not found" in progress.error.lower()


def test_start_preserves_message_for_unknown_planning_failure() -> None:
    def plan_fn(repo_id, repo_type):
        raise RuntimeError("disk full")

    mgr = DownloadManager(plan_fn=plan_fn, id_factory=lambda: "job1")
    job_id = mgr.start("meta/x")

    progress = mgr.progress(job_id)
    assert progress.state == "error"
    assert progress.error is not None and "disk full" in progress.error
