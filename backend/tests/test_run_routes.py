"""Tests for the /v1/runs routes (REST + WS) with a fake training service."""

import pytest
from fastapi.testclient import TestClient

from tinyforge.api.app import create_app
from tinyforge.services import Services
from tinyforge.train.models import RunRecord, RunStatus

TOKEN = "tok"


class FakeTraining:
    def __init__(self):
        self.calls: dict = {}
        self._state = "running"
        self._events = [
            {"event": "train", "iter": 1, "train_loss": 5.0},
            {"event": "val", "iter": 1, "val_loss": 4.5},
        ]

    def _record(self):
        return RunRecord(
            id="run1", name="exp", model_repo="m", dataset_id="ds1", state=self._state,
            created_at="t", adapter_path="/runs/run1", config={"iters": 5},
        )

    def start(self, request):
        self.calls["start"] = request
        return self._record()

    def list(self):
        return [self._record()]

    def get(self, run_id):
        return self._record()

    def status(self, run_id):
        return RunStatus(id=run_id, name="exp", state=self._state, num_events=len(self._events))

    def events(self, run_id, since=0):
        return self._events[since:]

    def stop(self, run_id):
        self.calls["stop"] = run_id
        self._state = "stopped"


@pytest.fixture
def client_and_training():
    training = FakeTraining()
    services = Services(
        auth=None, hub=None, downloads=None, cache=None, datasets=None, training=training
    )
    app = create_app(token=TOKEN, services=services)
    app.state.token = TOKEN
    return TestClient(app), training


def headers():
    return {"Authorization": f"Bearer {TOKEN}"}


def start_body():
    return {"name": "exp", "model_repo": "mlx-community/x", "dataset_id": "ds1", "iters": 5}


def test_start_requires_token(client_and_training) -> None:
    client, _ = client_and_training
    assert client.post("/v1/runs", json=start_body()).status_code == 401


def test_start_run_returns_record(client_and_training) -> None:
    client, training = client_and_training
    resp = client.post("/v1/runs", json=start_body(), headers=headers())
    assert resp.status_code == 200
    assert resp.json()["id"] == "run1"
    assert training.calls["start"].dataset_id == "ds1"


def test_list_and_get_and_events(client_and_training) -> None:
    client, _ = client_and_training
    assert client.get("/v1/runs", headers=headers()).json()[0]["name"] == "exp"
    assert client.get("/v1/runs/run1", headers=headers()).json()["id"] == "run1"
    events = client.get("/v1/runs/run1/events", params={"since": 1}, headers=headers()).json()
    assert events == [{"event": "val", "iter": 1, "val_loss": 4.5}]


def test_stop_run(client_and_training) -> None:
    client, training = client_and_training
    assert client.post("/v1/runs/run1/stop", headers=headers()).json()["ok"] is True
    assert training.calls["stop"] == "run1"


def test_ws_streams_events_then_terminal_status(client_and_training) -> None:
    client, training = client_and_training
    training._state = "completed"  # so the stream terminates promptly
    with client.websocket_connect(f"/v1/runs/run1/ws?token={TOKEN}") as ws:
        first = ws.receive_json()
        assert first["event"] == "train"
        # drain until the terminal status arrives
        messages = [first]
        for _ in range(5):
            messages.append(ws.receive_json())
            if messages[-1].get("event") == "status":
                break
    assert any(m.get("event") == "status" and m["state"] == "completed" for m in messages)


def test_ws_rejects_bad_token(client_and_training) -> None:
    client, _ = client_and_training
    with pytest.raises(Exception):
        with client.websocket_connect("/v1/runs/run1/ws?token=wrong") as ws:
            ws.receive_json()
