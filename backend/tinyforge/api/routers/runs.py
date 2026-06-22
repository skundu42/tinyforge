"""Training run routes: start, list, status, events (+ WS stream), stop."""

from __future__ import annotations

import asyncio
import hmac

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from tinyforge.train.models import RunRecord, RunStatus, StartRunRequest

router = APIRouter(prefix="/v1/runs")
ws_router = APIRouter()


def training_of(request: Request):
    return request.app.state.services.training


@router.post("")
def start_run(request: Request, body: StartRunRequest) -> RunRecord:
    return training_of(request).start(body)


@router.get("")
def list_runs(request: Request) -> list[RunRecord]:
    return training_of(request).list()


@router.get("/{run_id}")
def get_run(request: Request, run_id: str) -> RunRecord:
    return training_of(request).get(run_id)


@router.get("/{run_id}/status")
def run_status(request: Request, run_id: str) -> RunStatus:
    return training_of(request).status(run_id)


@router.get("/{run_id}/events")
def run_events(request: Request, run_id: str, since: int = 0) -> list[dict]:
    return training_of(request).events(run_id, since)


@router.post("/{run_id}/stop")
def stop_run(request: Request, run_id: str) -> dict:
    training_of(request).stop(run_id)
    return {"ok": True}


@ws_router.websocket("/v1/runs/{run_id}/ws")
async def run_ws(websocket: WebSocket, run_id: str) -> None:
    supplied = websocket.query_params.get("token", "")
    if not hmac.compare_digest(supplied, websocket.app.state.token):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    service = websocket.app.state.services.training
    cursor = 0
    try:
        while True:
            try:
                new_events = service.events(run_id, since=cursor)
            except KeyError:
                await websocket.close(code=1011)
                return
            for event in new_events:
                await websocket.send_json(event)
            cursor += len(new_events)

            status = service.status(run_id)
            if status.state in ("completed", "failed", "stopped"):
                await websocket.send_json(
                    {"event": "status", "state": status.state, "error": status.error}
                )
                break
            await asyncio.sleep(0.4)
    except WebSocketDisconnect:
        return
    await websocket.close()
