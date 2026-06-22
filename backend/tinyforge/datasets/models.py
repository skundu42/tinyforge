"""Pydantic models for the dataset builder."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class DatasetSource(BaseModel):
    kind: Literal["hub", "local"]
    repo_id: str | None = None  # hub
    config: str | None = None  # hub config/subset
    path: str | None = None  # local file path
    file_format: str | None = None  # json | csv | parquet (local)
    split: str = "train"


class DatasetPreview(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
    num_rows: int


class HistogramBin(BaseModel):
    lo: int
    hi: int
    count: int


class TokenStats(BaseModel):
    count: int
    min: int = 0
    max: int = 0
    mean: float = 0.0
    p50: int = 0
    p95: int = 0
    histogram: list[HistogramBin] = []


class RegisteredDataset(BaseModel):
    id: str
    name: str
    target_format: str  # text | completion | chat
    train_rows: int
    val_rows: int
    created_at: str
    path: str


class FormatSpec(BaseModel):
    mode: Literal["text", "prompt_completion", "messages", "alpaca"]
    text_column: str = "text"
    prompt_column: str = "prompt"
    completion_column: str = "completion"
    messages_column: str = "messages"
    instruction_column: str = "instruction"
    input_column: str = "input"
    output_column: str = "output"
