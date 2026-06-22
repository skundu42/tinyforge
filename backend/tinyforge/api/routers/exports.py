"""Export routes: start an export (fuse/convert/push) + poll status."""

from __future__ import annotations

from fastapi import APIRouter, Request

from tinyforge.export.models import ExportRequest, ExportStatus

router = APIRouter(prefix="/v1/exports")


def exports_of(request: Request):
    return request.app.state.services.exports


@router.post("")
def start_export(request: Request, body: ExportRequest) -> ExportStatus:
    manager = exports_of(request)
    job_id = manager.start(body)
    return manager.status(job_id)


@router.get("")
def list_exports(request: Request) -> list[ExportStatus]:
    return exports_of(request).list()


@router.get("/{export_id}")
def get_export(request: Request, export_id: str) -> ExportStatus:
    return exports_of(request).status(export_id)
