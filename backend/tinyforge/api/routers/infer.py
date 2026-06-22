"""Inference route: a WebSocket that streams generated tokens."""

from __future__ import annotations

import asyncio
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

    def produce() -> None:
        try:
            for delta in service.stream(request):
                loop.call_soon_threadsafe(queue.put_nowait, ("token", delta))
        except Exception as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

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
    except WebSocketDisconnect:
        return
    await websocket.close()
