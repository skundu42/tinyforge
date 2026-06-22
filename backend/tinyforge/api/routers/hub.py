"""HuggingFace Hub routes: search, detail, downloads (+ WS progress), cache, auth."""

from __future__ import annotations

import asyncio
import hmac

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from tinyforge.hub.models import (
    AuthStatus,
    CacheInfo,
    DownloadProgress,
    HubDataset,
    HubModel,
    HubModelDetail,
)
from tinyforge.services import Services

router = APIRouter(prefix="/v1/hub")
ws_router = APIRouter(prefix="/v1/hub")


def services_of(request: Request) -> Services:
    return request.app.state.services


class DownloadRequest(BaseModel):
    repo_id: str
    repo_type: str = "model"


class LoginRequest(BaseModel):
    token: str


@router.get("/models")
def search_models(
    request: Request,
    query: str | None = None,
    sort: str = "downloads",
    limit: int = 30,
    author: str | None = None,
    pipeline_tag: str | None = None,
    filter: str | None = None,
    gated: bool | None = None,
) -> list[HubModel]:
    return services_of(request).hub.search_models(
        query=query, sort=sort, limit=limit, author=author,
        pipeline_tag=pipeline_tag, filter=filter, gated=gated,
    )


@router.get("/datasets")
def search_datasets(
    request: Request,
    query: str | None = None,
    sort: str = "downloads",
    limit: int = 30,
    author: str | None = None,
) -> list[HubDataset]:
    return services_of(request).hub.search_datasets(
        query=query, sort=sort, limit=limit, author=author
    )


@router.get("/models/{repo_id:path}")
def model_detail(request: Request, repo_id: str) -> HubModelDetail:
    return services_of(request).hub.model_detail(repo_id)


@router.post("/downloads")
def start_download(request: Request, body: DownloadRequest) -> DownloadProgress:
    downloads = services_of(request).downloads
    job_id = downloads.start(body.repo_id, body.repo_type)
    return downloads.progress(job_id)


@router.get("/downloads/{job_id}")
def download_progress(request: Request, job_id: str) -> DownloadProgress:
    return services_of(request).downloads.progress(job_id)


@router.get("/cache")
def cache_info(request: Request) -> CacheInfo:
    return services_of(request).cache.info()


@router.delete("/cache/{repo_id:path}")
def delete_cached(request: Request, repo_id: str, repo_type: str | None = None) -> dict:
    freed = services_of(request).cache.delete(repo_id, repo_type)
    return {"freed_bytes": freed}


@router.get("/auth")
def auth_status(request: Request) -> AuthStatus:
    return services_of(request).auth.status()


@router.post("/auth/login")
def auth_login(request: Request, body: LoginRequest) -> AuthStatus:
    return services_of(request).auth.login(body.token)


@router.post("/auth/logout")
def auth_logout(request: Request) -> dict:
    services_of(request).auth.logout()
    return {"ok": True}


@ws_router.websocket("/downloads/{job_id}/ws")
async def download_ws(websocket: WebSocket, job_id: str) -> None:
    # WebSockets can't use the HTTP bearer dependency cleanly; validate the
    # per-launch token from the query string against the same secret.
    supplied = websocket.query_params.get("token", "")
    if not hmac.compare_digest(supplied, websocket.app.state.token):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    downloads = websocket.app.state.services.downloads
    try:
        while True:
            try:
                progress = downloads.progress(job_id)
            except KeyError:
                await websocket.close(code=1011)
                return
            await websocket.send_json(progress.model_dump())
            if progress.state in ("completed", "error"):
                break
            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        return
    await websocket.close()
