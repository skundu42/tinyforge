"""The app must reap child processes on graceful shutdown (SIGTERM path)."""

from fastapi.testclient import TestClient

import tinyforge.children as children
from tinyforge.api.app import create_app
from tinyforge.services import Services


def _empty_services() -> Services:
    return Services(
        auth=None, hub=None, downloads=None, cache=None, datasets=None,
        training=None, inference=None, exports=None,
    )


def test_graceful_shutdown_reaps_children(monkeypatch) -> None:
    calls = {"n": 0}
    monkeypatch.setattr(
        children.child_registry, "terminate_all",
        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1),
    )

    app = create_app(token="tok", services=_empty_services())
    with TestClient(app):  # entering runs startup, exiting runs shutdown
        pass

    assert calls["n"] >= 1
