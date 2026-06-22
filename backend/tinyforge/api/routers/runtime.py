"""Protected runtime-info endpoint (requires token)."""

from __future__ import annotations

from fastapi import APIRouter

from tinyforge import system

router = APIRouter(prefix="/v1")


@router.get("/runtime")
def runtime() -> dict[str, object]:
    return system.runtime_info()
