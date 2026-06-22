"""Dataset builder routes: preview, analyze (token stats), prepare, list, delete."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from tinyforge.datasets.models import (
    DatasetPreview,
    DatasetSource,
    FormatSpec,
    RegisteredDataset,
    TokenStats,
)

router = APIRouter(prefix="/v1/datasets")


def datasets_of(request: Request):
    return request.app.state.services.datasets


class PreviewRequest(BaseModel):
    source: DatasetSource
    limit: int = 20


class AnalyzeRequest(BaseModel):
    source: DatasetSource
    spec: FormatSpec
    tokenizer_repo: str
    sample: int = 200


class PrepareRequest(BaseModel):
    name: str
    source: DatasetSource
    spec: FormatSpec
    val_fraction: float = 0.1
    seed: int = 0
    max_rows: int | None = None


@router.post("/preview")
def preview(request: Request, body: PreviewRequest) -> DatasetPreview:
    return datasets_of(request).preview(body.source, body.limit)


@router.post("/analyze")
def analyze(request: Request, body: AnalyzeRequest) -> TokenStats:
    return datasets_of(request).analyze(body.source, body.spec, body.tokenizer_repo, body.sample)


@router.post("/prepare")
def prepare(request: Request, body: PrepareRequest) -> RegisteredDataset:
    return datasets_of(request).prepare(
        body.name, body.source, body.spec, body.val_fraction, body.seed, body.max_rows
    )


@router.get("")
def list_datasets(request: Request) -> list[RegisteredDataset]:
    return datasets_of(request).list()


@router.get("/{dataset_id}")
def get_dataset(request: Request, dataset_id: str) -> RegisteredDataset:
    return datasets_of(request).get(dataset_id)


@router.delete("/{dataset_id}")
def delete_dataset(request: Request, dataset_id: str) -> dict:
    datasets_of(request).delete(dataset_id)
    return {"ok": True}
