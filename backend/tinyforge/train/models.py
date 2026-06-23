"""Models for training runs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


# (num_layers, hidden_size, num_heads, context_length)
LM_PRESETS: dict[str, tuple[int, int, int, int]] = {
    "tiny": (4, 128, 4, 256),
    "small": (6, 256, 8, 512),
    "medium": (8, 512, 8, 512),
}


def apply_preset(
    model_size: str, num_layers: int, hidden_size: int, num_heads: int, context_length: int
) -> tuple[int, int, int, int]:
    """Resolve LM dimensions: named presets win; 'custom' (or unknown) passes values through."""
    if model_size in LM_PRESETS:
        return LM_PRESETS[model_size]
    return (num_layers, hidden_size, num_heads, context_length)


class RunConfig(BaseModel):
    name: str
    model_repo: str
    data_dir: str  # dir with train.jsonl / valid.jsonl (a prepared dataset)
    adapter_path: str  # output dir for adapters + events.jsonl
    engine: Literal["mlx", "lm"] = "mlx"
    fine_tune_type: Literal["lora", "dora", "full"] = "lora"
    num_layers: int = 16
    batch_size: int = 1
    iters: int = 100
    learning_rate: float = 1e-5
    steps_per_report: int = 10
    steps_per_eval: int = 50
    val_batches: int = 1
    max_seq_length: int = 512
    grad_checkpoint: bool = True
    seed: int = 0
    model_size: str = "small"
    hidden_size: int = 256
    num_heads: int = 8
    vocab_size: int = 8000
    context_length: int = 512


class RunStatus(BaseModel):
    id: str
    name: str
    state: str  # pending | running | completed | failed | stopped
    error: str | None = None
    num_events: int = 0


class RunRecord(BaseModel):
    id: str
    name: str
    model_repo: str
    dataset_id: str
    state: str
    created_at: str
    adapter_path: str
    config: dict


class StartRunRequest(BaseModel):
    name: str
    model_repo: str = ""
    dataset_id: str = ""
    engine: Literal["mlx", "lm"] = "mlx"
    fine_tune_type: str = "lora"
    num_layers: int = 16
    batch_size: int = 1
    iters: int = 100
    learning_rate: float = 1e-5
    steps_per_report: int = 10
    steps_per_eval: int = 50
    max_seq_length: int = 512
    grad_checkpoint: bool = True
    seed: int = 0
    model_size: str = "small"
    hidden_size: int = 256
    num_heads: int = 8
    vocab_size: int = 8000
    context_length: int = 512
