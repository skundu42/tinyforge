"""Tests for the inference WebSocket route with a fake generator."""

import pytest
from fastapi.testclient import TestClient

from tinyforge.api.app import create_app
from tinyforge.infer.service import InferenceService
from tinyforge.services import Services

TOKEN = "tok"


def make_client(generate_fn):
    services = Services(
        auth=None, hub=None, downloads=None, cache=None, datasets=None,
        training=None, inference=InferenceService(generate_fn=generate_fn),
    )
    app = create_app(token=TOKEN, services=services)
    app.state.token = TOKEN
    return TestClient(app)


def request_body():
    return {"model_repo": "mlx-community/x", "prompt": "Hello"}


def test_infer_streams_tokens_then_done() -> None:
    client = make_client(lambda req: iter(["Hello", " world", "!"]))
    with client.websocket_connect(f"/v1/infer/ws?token={TOKEN}") as ws:
        ws.send_json(request_body())
        messages = []
        for _ in range(10):
            msg = ws.receive_json()
            messages.append(msg)
            if msg["event"] == "done":
                break
    tokens = [m["text"] for m in messages if m["event"] == "token"]
    assert tokens == ["Hello", " world", "!"]
    assert messages[-1]["event"] == "done"


def test_infer_forwards_request_params() -> None:
    captured = {}

    def fake(req):
        captured["repo"] = req.model_repo
        captured["adapter"] = req.adapter_path
        captured["temp"] = req.temp
        return iter(["ok"])

    client = make_client(fake)
    with client.websocket_connect(f"/v1/infer/ws?token={TOKEN}") as ws:
        ws.send_json({"model_repo": "m/x", "adapter_path": "/runs/r1", "prompt": "hi", "temp": 0.2})
        while ws.receive_json()["event"] != "done":
            pass
    assert captured == {"repo": "m/x", "adapter": "/runs/r1", "temp": 0.2}


def test_infer_reports_error() -> None:
    def boom(req):
        raise RuntimeError("model not found")
        yield  # pragma: no cover

    client = make_client(boom)
    with client.websocket_connect(f"/v1/infer/ws?token={TOKEN}") as ws:
        ws.send_json(request_body())
        msg = ws.receive_json()
        assert msg["event"] == "error"
        assert "model not found" in msg["error"]


def test_infer_rejects_bad_token() -> None:
    client = make_client(lambda req: iter(["x"]))
    with pytest.raises(Exception):
        with client.websocket_connect("/v1/infer/ws?token=wrong") as ws:
            ws.receive_json()
