"""Tests for the /v1/hub API routes (REST + WS) with fake services."""

import pytest
from fastapi.testclient import TestClient

from tinyforge.api.app import create_app
from tinyforge.hub.models import (
    AuthStatus,
    CachedRepo,
    CacheInfo,
    DownloadProgress,
    HubDataset,
    HubModel,
    HubModelDetail,
)
from tinyforge.services import Services

TOKEN = "tok"


class FakeHub:
    def __init__(self):
        self.calls: dict = {}

    def search_models(self, query=None, **kw):
        self.calls["search_models"] = {"query": query, **kw}
        return [HubModel(id="mlx-community/x-4bit", downloads=10)]

    def search_datasets(self, query=None, **kw):
        self.calls["search_datasets"] = {"query": query, **kw}
        return [HubDataset(id="rajpurkar/squad", downloads=5)]

    def model_detail(self, repo_id):
        return HubModelDetail(id=repo_id, readme="# R")


class FakeDownloads:
    def __init__(self):
        self._jobs: dict = {}

    def start(self, repo_id, repo_type="model"):
        self._jobs["job1"] = DownloadProgress(
            id="job1", repo_id=repo_id, repo_type=repo_type, total_bytes=100,
            downloaded_bytes=100, fraction=1.0, state="completed",
        )
        return "job1"

    def progress(self, job_id):
        return self._jobs[job_id]


class FakeCache:
    def info(self):
        return CacheInfo(
            size_on_disk=2000,
            repos=[CachedRepo(repo_id="a/x", repo_type="model", size_on_disk=2000, nb_files=2)],
        )

    def delete(self, repo_id, repo_type=None):
        return 2000


class FakeAuth:
    def status(self):
        return AuthStatus(logged_in=True, name="alice")

    def login(self, token):
        return AuthStatus(logged_in=True, name="alice")

    def logout(self):
        pass


@pytest.fixture
def client_and_services():
    services = Services(
        auth=FakeAuth(), hub=FakeHub(), downloads=FakeDownloads(), cache=FakeCache(), datasets=None
    )
    app = create_app(token=TOKEN, services=services)
    return TestClient(app), services


def headers():
    return {"Authorization": f"Bearer {TOKEN}"}


def test_search_models_requires_token(client_and_services) -> None:
    client, _ = client_and_services
    assert client.get("/v1/hub/models").status_code == 401


def test_search_models_returns_and_forwards_params(client_and_services) -> None:
    client, services = client_and_services
    resp = client.get("/v1/hub/models", params={"query": "llama", "limit": 5}, headers=headers())
    assert resp.status_code == 200
    assert resp.json()[0]["id"] == "mlx-community/x-4bit"
    assert services.hub.calls["search_models"]["query"] == "llama"
    assert services.hub.calls["search_models"]["limit"] == 5


def test_search_datasets_returns_results(client_and_services) -> None:
    client, _ = client_and_services
    resp = client.get("/v1/hub/datasets", params={"query": "squad"}, headers=headers())
    assert resp.json()[0]["id"] == "rajpurkar/squad"


def test_model_detail_with_slashed_repo_id(client_and_services) -> None:
    client, _ = client_and_services
    resp = client.get("/v1/hub/models/mlx-community/x-4bit", headers=headers())
    assert resp.status_code == 200
    assert resp.json()["readme"] == "# R"


def test_start_then_get_download_progress(client_and_services) -> None:
    client, _ = client_and_services
    started = client.post("/v1/hub/downloads", json={"repo_id": "mlx-community/x-4bit"}, headers=headers())
    assert started.status_code == 200
    job_id = started.json()["id"]
    progress = client.get(f"/v1/hub/downloads/{job_id}", headers=headers())
    assert progress.json()["state"] == "completed"
    assert progress.json()["fraction"] == 1.0


def test_cache_info_and_delete(client_and_services) -> None:
    client, _ = client_and_services
    assert client.get("/v1/hub/cache", headers=headers()).json()["size_on_disk"] == 2000
    deleted = client.request("DELETE", "/v1/hub/cache/a/x", headers=headers())
    assert deleted.json()["freed_bytes"] == 2000


def test_auth_status_login_logout(client_and_services) -> None:
    client, _ = client_and_services
    assert client.get("/v1/hub/auth", headers=headers()).json()["name"] == "alice"
    login = client.post("/v1/hub/auth/login", json={"token": "hf_x"}, headers=headers())
    assert login.json()["logged_in"] is True
    assert client.post("/v1/hub/auth/logout", headers=headers()).json()["ok"] is True


def test_download_ws_streams_progress_with_token(client_and_services) -> None:
    client, _ = client_and_services
    job_id = client.post("/v1/hub/downloads", json={"repo_id": "a/x"}, headers=headers()).json()["id"]
    with client.websocket_connect(f"/v1/hub/downloads/{job_id}/ws?token={TOKEN}") as ws:
        message = ws.receive_json()
        assert message["state"] == "completed"


def test_download_ws_rejects_bad_token(client_and_services) -> None:
    client, _ = client_and_services
    job_id = client.post("/v1/hub/downloads", json={"repo_id": "a/x"}, headers=headers()).json()["id"]
    with pytest.raises(Exception):
        with client.websocket_connect(f"/v1/hub/downloads/{job_id}/ws?token=wrong") as ws:
            ws.receive_json()
