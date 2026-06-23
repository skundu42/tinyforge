"""Threaded Hub download manager with progress tracking.

Total size comes from a dry-run. Progress comes from a byte-counting tqdm wired
into per-file `hf_hub_download` calls: `snapshot_download` hardcodes its own
aggregate tqdm and won't surface per-file bytes, and Xet (the default transfer
backend) materialises large files atomically — so neither `snapshot_download`
nor blobs-polling reports byte progress. Downloading file-by-file lets the
per-file byte bar reach our hook. The planner and downloader are injectable so
the job lifecycle is testable without network.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from tinyforge.hub.errors import classify_hub_error
from tinyforge.hub.models import DownloadPlan, DownloadPlanFile, DownloadProgress

# plan_fn(repo_id, repo_type) -> list of (filename, size, will_download)
PlanFn = Callable[[str, str], list[tuple[str, int, bool]]]
# download_fn(repo_id, repo_type, on_progress) -> local path
DownloadFn = Callable[[str, str, Callable[[int], None]], str]


def _friendly_download_error(exc: Exception) -> str:
    """Translate a planning/transfer failure into an actionable message.

    Planning runs synchronously in the request handler, so an uncaught failure
    surfaces to the client as an opaque HTTP 500. The two most common causes —
    a gated/private repo with no token, and a mistyped repo id — are worth
    spelling out; everything else passes through verbatim.
    """
    classified = classify_hub_error(exc)
    return classified[1] if classified else str(exc)


@dataclass
class _Job:
    id: str
    repo_id: str
    repo_type: str
    total_bytes: int
    downloaded_bytes: int = 0
    state: str = "pending"
    error: str | None = None
    local_path: str | None = None


def _default_plan_fn(repo_id: str, repo_type: str) -> list[tuple[str, int, bool]]:
    import huggingface_hub

    entries = huggingface_hub.snapshot_download(repo_id, repo_type=repo_type, dry_run=True)
    return [(e.filename, e.file_size or 0, bool(e.will_download)) for e in entries]


def _default_download_fn(
    repo_id: str, repo_type: str, on_progress: Callable[[int], None]
) -> str:
    import huggingface_hub
    from tqdm.auto import tqdm

    class _ByteTqdm(tqdm):
        def update(self, n=1):
            on_progress(int(n or 0))
            return super().update(n)

    entries = huggingface_hub.snapshot_download(repo_id, repo_type=repo_type, dry_run=True)
    for entry in entries:
        if entry.will_download:
            huggingface_hub.hf_hub_download(
                repo_id, entry.filename, repo_type=repo_type, tqdm_class=_ByteTqdm
            )
    # All files now cached; this resolves and returns the snapshot directory.
    return huggingface_hub.snapshot_download(repo_id, repo_type=repo_type)


class DownloadManager:
    def __init__(
        self,
        *,
        plan_fn: PlanFn = _default_plan_fn,
        download_fn: DownloadFn = _default_download_fn,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._plan_fn = plan_fn
        self._download_fn = download_fn
        self._id_factory = id_factory
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()

    def plan(self, repo_id: str, repo_type: str = "model") -> DownloadPlan:
        entries = self._plan_fn(repo_id, repo_type)
        files = [
            DownloadPlanFile(filename=name, size=size, will_download=will)
            for name, size, will in entries
        ]
        total = sum(f.size for f in files if f.will_download)
        cached = sum(f.size for f in files if not f.will_download)
        return DownloadPlan(
            repo_id=repo_id, repo_type=repo_type, files=files,
            total_bytes=total, already_cached_bytes=cached,
        )

    def start(self, repo_id: str, repo_type: str = "model") -> str:
        job_id = self._id_factory()
        try:
            plan = self.plan(repo_id, repo_type)
        except Exception as exc:  # noqa: BLE001 - recorded as an error job, not raised
            # Planning happens in the request handler; surfacing the failure as an
            # error job (rather than letting it escape as a bare HTTP 500) lets the
            # client show why the download could not start.
            job = _Job(
                id=job_id, repo_id=repo_id, repo_type=repo_type,
                total_bytes=0, state="error", error=_friendly_download_error(exc),
            )
            with self._lock:
                self._jobs[job_id] = job
            return job_id
        job = _Job(
            id=job_id, repo_id=repo_id, repo_type=repo_type,
            total_bytes=plan.total_bytes,
        )
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job.id

    def _run(self, job: _Job) -> None:
        with self._lock:
            job.state = "running"

        def on_progress(delta: int) -> None:
            with self._lock:
                updated = job.downloaded_bytes + delta
                job.downloaded_bytes = min(updated, job.total_bytes) if job.total_bytes else updated

        try:
            path = self._download_fn(job.repo_id, job.repo_type, on_progress)
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            with self._lock:
                job.state = "error"
                job.error = _friendly_download_error(exc)
            return

        with self._lock:
            job.state = "completed"
            job.local_path = path
            job.downloaded_bytes = job.total_bytes

    def progress(self, job_id: str) -> DownloadProgress:
        with self._lock:
            job = self._jobs[job_id]
            total = job.total_bytes
            downloaded = job.downloaded_bytes
            if total > 0:
                fraction = min(downloaded / total, 1.0)
            else:
                fraction = 1.0 if job.state == "completed" else 0.0
            return DownloadProgress(
                id=job.id, repo_id=job.repo_id, repo_type=job.repo_type,
                total_bytes=total, downloaded_bytes=downloaded, fraction=fraction,
                state=job.state, error=job.error, local_path=job.local_path,
            )
