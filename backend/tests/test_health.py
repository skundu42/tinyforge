"""Tests for the FastAPI app: liveness health check + bearer-token auth."""

import pytest
from fastapi.testclient import TestClient

from tinyforge.api.app import create_app

TOKEN = "test-token-abc123"


@pytest.fixture
def client() -> TestClient:
    app = create_app(token=TOKEN)
    return TestClient(app)


def test_health_is_open_and_reports_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["name"] == "tinyforge"
    assert "version" in body


def test_runtime_requires_token_when_missing(client: TestClient) -> None:
    resp = client.get("/v1/runtime")
    assert resp.status_code == 401


def test_runtime_rejects_wrong_token(client: TestClient) -> None:
    resp = client.get("/v1/runtime", headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 401


def test_runtime_accepts_correct_token(client: TestClient) -> None:
    resp = client.get("/v1/runtime", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["python_version"].startswith("3.")
    assert body["platform"] == "darwin"
    assert "engines" in body
