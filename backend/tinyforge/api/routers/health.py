"""Open liveness/readiness endpoint (no auth)."""

from __future__ import annotations

from fastapi import APIRouter

from tinyforge import __version__

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "name": "tinyforge", "version": __version__}
