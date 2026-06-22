"""FastAPI application factory.

`create_app(token)` builds the full API. The health router is public; every
`/v1` router is guarded by the per-launch bearer token.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI

from tinyforge import __version__
from tinyforge.api.auth import make_token_dependency
from tinyforge.api.routers import health, runtime


def create_app(token: str) -> FastAPI:
    app = FastAPI(title="TinyForge backend", version=__version__)

    require_token = make_token_dependency(token)

    # Public liveness probe.
    app.include_router(health.router)

    # Everything under /v1 requires the token.
    app.include_router(runtime.router, dependencies=[Depends(require_token)])

    return app
