"""Turn a prepared text dataset into packed token blocks for from-scratch LM training.

Reads the `train.jsonl` / `valid.jsonl` the dataset builder writes (text / prompt-
completion / messages rows), renders each row to plain text, tokenizes, concatenates
documents with an EOS separator, and chunks into fixed-length blocks for causal LM.
"""

from __future__ import annotations

import json
import os
from typing import Any


def render_text(row: dict[str, Any]) -> str:
    """Render one prepared dataset row to a single training string."""
    if "text" in row:
        return str(row["text"])
    if "prompt" in row and "completion" in row:
        return f"{row['prompt']}\n\n{row['completion']}"
    if "messages" in row:
        return "\n".join(f"{m['role']}: {m['content']}" for m in row["messages"])
    raise ValueError(f"row has no recognized text fields: {sorted(row)}")


def load_corpus(data_dir: str) -> tuple[list[str], list[str]]:
    """Read train/valid jsonl from a prepared dataset dir into lists of rendered text."""
    def _read(name: str) -> list[str]:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as handle:
            return [render_text(json.loads(line)) for line in handle if line.strip()]

    return _read("train.jsonl"), _read("valid.jsonl")


def pack_tokens(token_lists: list[list[int]], block_size: int, eos_id: int) -> list[list[int]]:
    """Concatenate token lists (EOS between docs) and split into full `block_size` blocks."""
    stream: list[int] = []
    for ids in token_lists:
        stream.extend(ids)
        stream.append(eos_id)
    n_blocks = len(stream) // block_size
    return [stream[i * block_size : (i + 1) * block_size] for i in range(n_blocks)]


class PackedTextDataset:
    """A torch Dataset of fixed-length blocks; labels == input_ids for causal LM."""

    def __init__(self, blocks: list[list[int]]) -> None:
        self._blocks = blocks

    def __len__(self) -> int:
        return len(self._blocks)

    def __getitem__(self, i: int) -> dict:
        import torch

        ids = torch.tensor(self._blocks[i], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}
