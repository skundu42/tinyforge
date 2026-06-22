"""FastAPI application factory.

`create_app(token)` builds the full API. The health router is public; every
`/v1` router is guarded by the per-launch bearer token.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI

from tinyforge import __version__
from tinyforge.api.auth import make_token_dependency
from tinyforge.api.routers import datasets, health, hub, infer, runs, runtime
from tinyforge.services import Services, build_services


def create_app(token: str, services: Services | None = None) -> FastAPI:
    app = FastAPI(title="TinyForge backend", version=__version__)

    app.state.token = token
    app.state.services = services if services is not None else build_services()

    require_token = make_token_dependency(token)

    # Public liveness probe.
    app.include_router(health.router)

    # Everything under /v1 requires the token.
    app.include_router(runtime.router, dependencies=[Depends(require_token)])
    app.include_router(hub.router, dependencies=[Depends(require_token)])
    app.include_router(datasets.router, dependencies=[Depends(require_token)])
    app.include_router(runs.router, dependencies=[Depends(require_token)])

    # WebSocket routes validate the token from the query string internally.
    app.include_router(hub.ws_router)
    app.include_router(runs.ws_router)
    app.include_router(infer.ws_router)

    return app
