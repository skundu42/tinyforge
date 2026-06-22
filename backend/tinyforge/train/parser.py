"""Parse mlx_lm.lora training output lines into structured events."""

from __future__ import annotations

import re

_TRAIN = re.compile(
    r"Iter (?P<iter>\d+): Train loss (?P<loss>[\d.]+), "
    r"Learning Rate (?P<lr>[\d.eE+-]+), It/sec (?P<its>[\d.]+), "
    r"Tokens/sec (?P<tps>[\d.]+), Trained Tokens (?P<tt>\d+), "
    r"Peak mem (?P<mem>[\d.]+) GB"
)
_VAL = re.compile(r"Iter (?P<iter>\d+): Val loss (?P<loss>[\d.]+)")
_SAVED = re.compile(r"Saved.*weights to (?P<path>.+?)\.?\s*$")
_PARAMS = re.compile(r"Trainable parameters:")


def parse_line(line: str) -> dict | None:
    """Return a structured event for a recognized line, else None."""
    text = line.strip()
    if not text:
        return None

    if match := _TRAIN.search(text):
        return {
            "event": "train",
            "iter": int(match["iter"]),
            "train_loss": float(match["loss"]),
            "lr": float(match["lr"]),
            "it_per_sec": float(match["its"]),
            "tokens_per_sec": float(match["tps"]),
            "trained_tokens": int(match["tt"]),
            "peak_mem_gb": float(match["mem"]),
        }
    if match := _VAL.search(text):
        return {"event": "val", "iter": int(match["iter"]), "val_loss": float(match["loss"])}
    if match := _SAVED.search(text):
        return {"event": "saved", "path": match["path"].strip()}
    if _PARAMS.search(text):
        return {"event": "info", "text": text}
    return None
