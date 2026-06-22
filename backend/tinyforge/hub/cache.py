"""Local Hub cache inspection and deletion."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tinyforge.hub.models import CachedRepo, CacheInfo


def _default_scan() -> Any:
    import huggingface_hub

    return huggingface_hub.scan_cache_dir()


def scan_cache(scan_fn: Callable[[], Any] = _default_scan) -> CacheInfo:
    info = scan_fn()
    repos = [
        CachedRepo(
            repo_id=repo.repo_id,
            repo_type=str(repo.repo_type),
            size_on_disk=repo.size_on_disk,
            nb_files=repo.nb_files,
            last_accessed=getattr(repo, "last_accessed", None),
        )
        for repo in info.repos
    ]
    repos.sort(key=lambda r: r.size_on_disk, reverse=True)
    return CacheInfo(size_on_disk=info.size_on_disk, repos=repos)


def delete_repo(
    repo_id: str,
    repo_type: str | None = None,
    scan_fn: Callable[[], Any] = _default_scan,
) -> int:
    """Delete all cached revisions of a repo. Returns bytes freed (0 if absent)."""
    info = scan_fn()
    hashes = [
        rev.commit_hash
        for repo in info.repos
        if repo.repo_id == repo_id
        and (repo_type is None or str(repo.repo_type) == repo_type)
        for rev in repo.revisions
    ]
    if not hashes:
        return 0
    strategy = info.delete_revisions(*hashes)
    freed = strategy.expected_freed_size
    strategy.execute()
    return freed


class CacheService:
    """Thin object wrapper so routes can depend on an injectable cache service."""

    def __init__(self, scan_fn: Callable[[], Any] = _default_scan) -> None:
        self._scan_fn = scan_fn

    def info(self) -> CacheInfo:
        return scan_cache(scan_fn=self._scan_fn)

    def delete(self, repo_id: str, repo_type: str | None = None) -> int:
        return delete_repo(repo_id, repo_type, scan_fn=self._scan_fn)
