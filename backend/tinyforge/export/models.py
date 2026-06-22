"""Models for model exports."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ExportRequest(BaseModel):
    run_id: str
    target: Literal["safetensors", "mlx", "gguf"] = "safetensors"
    q_bits: int = 4
    push_repo: str | None = None  # if set, push the result to this HF repo


class ExportStatus(BaseModel):
    id: str
    run_id: str
    target: str
    state: str  # running | completed | failed
    error: str | None = None
    output_path: str | None = None
    hub_url: str | None = None
