"""Inference route: a WebSocket that streams generated tokens."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import threading

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from tinyforge.infer.service import GenRequest

ws_router = APIRouter()


@ws_router.websocket("/v1/infer/ws")
async def infer_ws(websocket: WebSocket) -> None:
    supplied = websocket.query_params.get("token", "")
    if not hmac.compare_digest(supplied, websocket.app.state.token):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    service = websocket.app.state.services.inference

    try:
        request = GenRequest(**(await websocket.receive_json()))
    except (WebSocketDisconnect, Exception):  # noqa: BLE001
        await websocket.close(code=1003)
        return

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    # Set when the client goes away (or we finish): tells the producer thread to
    # stop generating so a closed Playground doesn't keep the GPU busy.
    cancel = threading.Event()

    def post(item: tuple) -> None:
        # The handler may have already returned (loop closed); dropping the item
        # is fine since nothing is listening.
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(queue.put_nowait, item)

    def produce() -> None:
        try:
            for delta in service.stream(request, should_cancel=cancel.is_set):
                post(("token", delta))
        except Exception as exc:  # noqa: BLE001
            post(("error", str(exc)))
        post(("done", None))

    threading.Thread(target=produce, daemon=True).start()

    try:
        while True:
            kind, payload = await queue.get()
            if kind == "token":
                await websocket.send_json({"event": "token", "text": payload})
            elif kind == "error":
                await websocket.send_json({"event": "error", "error": payload})
                break
            else:
                await websocket.send_json({"event": "done"})
                break
    except Exception:  # noqa: BLE001 - client disconnected / send failed; stop generating
        pass
    finally:
        cancel.set()
    with contextlib.suppress(Exception):
        await websocket.close()
