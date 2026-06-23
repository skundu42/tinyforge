"""FastAPI application factory.

`create_app(token)` builds the full API. The health router is public; every
`/v1` router is guarded by the per-launch bearer token.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from tinyforge import __version__
from tinyforge.api.auth import make_token_dependency
from tinyforge.api.routers import datasets, exports, health, hub, infer, runs, runtime
from tinyforge.services import Services, build_services


@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield
    # Graceful shutdown (e.g. SIGTERM from the host app): reap any training/export
    # children so they don't outlive the backend and hold the GPU.
    from tinyforge.children import child_registry

    child_registry.terminate_all()


def create_app(token: str, services: Services | None = None) -> FastAPI:
    app = FastAPI(title="TinyForge backend", version=__version__, lifespan=_lifespan)

    app.state.token = token
    app.state.services = services if services is not None else build_services()

    @app.exception_handler(KeyError)
    async def _missing_resource(request: Request, exc: KeyError) -> JSONResponse:
        # Registries raise a bare KeyError(id) for an unknown run/dataset/export;
        # surface that as 404 instead of an opaque 500.
        key = exc.args[0] if exc.args else "resource"
        return JSONResponse(status_code=404, content={"detail": f"{key} not found"})

    require_token = make_token_dependency(token)

    # Public liveness probe.
    app.include_router(health.router)

    # Everything under /v1 requires the token.
    app.include_router(runtime.router, dependencies=[Depends(require_token)])
    app.include_router(hub.router, dependencies=[Depends(require_token)])
    app.include_router(datasets.router, dependencies=[Depends(require_token)])
    app.include_router(runs.router, dependencies=[Depends(require_token)])
    app.include_router(exports.router, dependencies=[Depends(require_token)])

    # WebSocket routes validate the token from the query string internally.
    app.include_router(hub.ws_router)
    app.include_router(runs.ws_router)
    app.include_router(infer.ws_router)

    return app
