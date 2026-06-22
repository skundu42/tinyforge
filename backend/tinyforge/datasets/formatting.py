"""Map raw dataset rows into the formats mlx-lm consumes.

Target formats:
  * "text"       -> {"text": ...}
  * "completion" -> {"prompt": ..., "completion": ...}
  * "chat"       -> {"messages": [{"role","content"}, ...]}
"""

from __future__ import annotations

from typing import Any

from tinyforge.datasets.models import FormatSpec

_TARGET = {
    "text": "text",
    "prompt_completion": "completion",
    "alpaca": "completion",
    "messages": "chat",
}


def target_format(spec: FormatSpec) -> str:
    """The mlx-lm dataset format this spec produces."""
    return _TARGET[spec.mode]


def format_row(row: dict[str, Any], spec: FormatSpec) -> dict[str, Any]:
    if spec.mode == "text":
        return {"text": row[spec.text_column]}
    if spec.mode == "prompt_completion":
        return {"prompt": row[spec.prompt_column], "completion": row[spec.completion_column]}
    if spec.mode == "messages":
        return {"messages": row[spec.messages_column]}
    if spec.mode == "alpaca":
        instruction = row[spec.instruction_column]
        extra = row.get(spec.input_column, "")
        prompt = f"{instruction}\n\n{extra}" if extra else instruction
        return {"prompt": prompt, "completion": row[spec.output_column]}
    raise ValueError(f"unknown mode: {spec.mode}")  # pragma: no cover


def format_rows(rows: list[dict[str, Any]], spec: FormatSpec) -> list[dict[str, Any]]:
    return [format_row(row, spec) for row in rows]
