"""Tests for the /v1/exports routes with a fake export manager."""

import pytest
from fastapi.testclient import TestClient

from tinyforge.api.app import create_app
from tinyforge.export.models import ExportStatus
from tinyforge.services import Services

TOKEN = "tok"


class FakeExports:
    def __init__(self):
        self.started = None

    def start(self, request):
        self.started = request
        return "exp1"

    def status(self, export_id):
        return ExportStatus(
            id=export_id, run_id="r1", target="safetensors", state="completed",
            output_path="/exports/exp1/fused", hub_url=None,
        )

    def list(self):
        return [self.status("exp1")]


@pytest.fixture
def client_and_exports():
    exports = FakeExports()
    services = Services(
        auth=None, hub=None, downloads=None, cache=None, datasets=None,
        training=None, inference=None, exports=exports,
    )
    return TestClient(create_app(token=TOKEN, services=services)), exports


def headers():
    return {"Authorization": f"Bearer {TOKEN}"}


def test_start_requires_token(client_and_exports) -> None:
    client, _ = client_and_exports
    assert client.post("/v1/exports", json={"run_id": "r1"}).status_code == 401


def test_start_export_returns_status(client_and_exports) -> None:
    client, exports = client_and_exports
    resp = client.post(
        "/v1/exports", json={"run_id": "r1", "target": "mlx", "q_bits": 4}, headers=headers()
    )
    assert resp.status_code == 200
    assert resp.json()["id"] == "exp1"
    assert exports.started.target == "mlx"


def test_list_and_get(client_and_exports) -> None:
    client, _ = client_and_exports
    assert client.get("/v1/exports", headers=headers()).json()[0]["id"] == "exp1"
    assert client.get("/v1/exports/exp1", headers=headers()).json()["state"] == "completed"


def test_get_unknown_export_returns_404(client_and_exports) -> None:
    client, exports = client_and_exports

    def missing(export_id):
        raise KeyError(export_id)

    exports.status = missing
    assert client.get("/v1/exports/nope", headers=headers()).status_code == 404
