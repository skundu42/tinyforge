"""Tests for the /v1/datasets routes with a fake dataset service."""

import pytest
from fastapi.testclient import TestClient

from tinyforge.api.app import create_app
from tinyforge.datasets.models import DatasetPreview, RegisteredDataset, TokenStats
from tinyforge.services import Services

TOKEN = "tok"


class FakeDatasets:
    def __init__(self):
        self.calls: dict = {}

    def preview(self, source, limit=20):
        self.calls["preview"] = {"source": source, "limit": limit}
        return DatasetPreview(columns=["text"], rows=[{"text": "hi"}], num_rows=1)

    def analyze(self, source, spec, tokenizer_repo, sample=200, token=None):
        return TokenStats(count=3, min=1, max=9, mean=5.0, p50=5, p95=9, histogram=[])

    def prepare(self, name, source, spec, val_fraction=0.1, seed=0, max_rows=None):
        self.calls["prepare"] = {"name": name, "val_fraction": val_fraction}
        return RegisteredDataset(
            id="ds1", name=name, target_format="completion", train_rows=9, val_rows=1,
            created_at="t", path="/data/ds1",
        )

    def list(self):
        return [RegisteredDataset(
            id="ds1", name="math", target_format="completion", train_rows=9, val_rows=1,
            created_at="t", path="/data/ds1")]

    def get(self, dataset_id):
        return self.list()[0]

    def delete(self, dataset_id):
        self.calls["delete"] = dataset_id


@pytest.fixture
def client_and_services():
    services = Services(
        auth=None, hub=None, downloads=None, cache=None, datasets=FakeDatasets(),
        training=None, inference=None,
    )
    return TestClient(create_app(token=TOKEN, services=services)), services


def headers():
    return {"Authorization": f"Bearer {TOKEN}"}


def hub_source():
    return {"kind": "hub", "repo_id": "org/ds", "split": "train"}


def test_preview_requires_token(client_and_services) -> None:
    client, _ = client_and_services
    assert client.post("/v1/datasets/preview", json={"source": hub_source()}).status_code == 401


def test_preview_returns_columns_and_rows(client_and_services) -> None:
    client, _ = client_and_services
    resp = client.post("/v1/datasets/preview", json={"source": hub_source(), "limit": 5}, headers=headers())
    assert resp.status_code == 200
    assert resp.json()["columns"] == ["text"]


def test_analyze_returns_token_stats(client_and_services) -> None:
    client, _ = client_and_services
    body = {"source": hub_source(), "spec": {"mode": "text"}, "tokenizer_repo": "org/m"}
    resp = client.post("/v1/datasets/analyze", json=body, headers=headers())
    assert resp.json()["max"] == 9


def test_prepare_registers_dataset(client_and_services) -> None:
    client, services = client_and_services
    body = {"name": "math", "source": hub_source(), "spec": {"mode": "alpaca"}, "val_fraction": 0.2}
    resp = client.post("/v1/datasets/prepare", json=body, headers=headers())
    assert resp.json()["id"] == "ds1"
    assert services.datasets.calls["prepare"]["val_fraction"] == 0.2


def test_list_and_delete(client_and_services) -> None:
    client, services = client_and_services
    assert client.get("/v1/datasets", headers=headers()).json()[0]["name"] == "math"
    assert client.request("DELETE", "/v1/datasets/ds1", headers=headers()).json()["ok"] is True
    assert services.datasets.calls["delete"] == "ds1"
