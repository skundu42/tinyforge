"""Tests for cache scan/delete mapping (scan_cache_dir behind a fake)."""

from types import SimpleNamespace

from tinyforge.hub.cache import delete_repo, scan_cache


def make_repo(repo_id, repo_type="model", size=1000, nb_files=3, revisions=None):
    return SimpleNamespace(
        repo_id=repo_id, repo_type=repo_type, size_on_disk=size, nb_files=nb_files,
        last_accessed=123.0, revisions=revisions or [],
    )


def test_scan_cache_maps_and_sorts_by_size_desc() -> None:
    info = SimpleNamespace(
        size_on_disk=3000, warnings=[],
        repos=[make_repo("a/small", size=1000), make_repo("b/big", size=2000)],
    )
    result = scan_cache(scan_fn=lambda: info)

    assert result.size_on_disk == 3000
    assert [r.repo_id for r in result.repos] == ["b/big", "a/small"]
    assert result.repos[0].size_on_disk == 2000


def test_delete_repo_collects_revisions_and_executes() -> None:
    repo = make_repo(
        "a/x", revisions=[SimpleNamespace(commit_hash="h1"), SimpleNamespace(commit_hash="h2")]
    )
    state: dict = {}

    class Strategy:
        expected_freed_size = 555

        def execute(self):
            state["executed"] = True

    class Info:
        size_on_disk = 1000
        warnings: list = []
        repos = [repo]

        def delete_revisions(self, *hashes):
            state["hashes"] = hashes
            return Strategy()

    freed = delete_repo("a/x", scan_fn=lambda: Info())

    assert freed == 555
    assert state["executed"] is True
    assert set(state["hashes"]) == {"h1", "h2"}


def test_delete_repo_unknown_returns_zero_without_executing() -> None:
    info = SimpleNamespace(size_on_disk=0, warnings=[], repos=[])
    assert delete_repo("missing/x", scan_fn=lambda: info) == 0
