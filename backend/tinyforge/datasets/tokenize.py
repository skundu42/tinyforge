"""Render formatted rows to text and compute token-length statistics.

`length_fn` is injectable so the statistics are testable without a tokenizer;
`make_length_fn` builds the real one from a Hub tokenizer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tinyforge.datasets.models import HistogramBin, TokenStats


def render_for_tokenization(row: dict[str, Any], target: str) -> str:
    if target == "text":
        return str(row.get("text", ""))
    if target == "completion":
        return f"{row.get('prompt', '')}{row.get('completion', '')}"
    if target == "chat":
        messages = row.get("messages", [])
        return "\n".join(str(m.get("content", "")) for m in messages)
    return ""


def _nearest_rank(sorted_values: list[int], quantile: float) -> int:
    if not sorted_values:
        return 0
    index = min(int(quantile * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[index]


def compute_token_stats(
    texts: list[str], length_fn: Callable[[str], int], bins: int = 12
) -> TokenStats:
    lengths = [length_fn(text) for text in texts]
    if not lengths:
        return TokenStats(count=0)

    ordered = sorted(lengths)
    low, high = ordered[0], ordered[-1]
    histogram = _histogram(lengths, low, high, bins)

    return TokenStats(
        count=len(lengths),
        min=low,
        max=high,
        mean=sum(lengths) / len(lengths),
        p50=_nearest_rank(ordered, 0.50),
        p95=_nearest_rank(ordered, 0.95),
        histogram=histogram,
    )


def _histogram(lengths: list[int], low: int, high: int, bins: int) -> list[HistogramBin]:
    if high == low:
        return [HistogramBin(lo=low, hi=high, count=len(lengths))]
    width = (high - low) / bins
    counts = [0] * bins
    for length in lengths:
        index = min(int((length - low) / width), bins - 1)
        counts[index] += 1
    return [
        HistogramBin(lo=round(low + i * width), hi=round(low + (i + 1) * width), count=count)
        for i, count in enumerate(counts)
    ]


def make_length_fn(repo_id: str, token: str | None = None) -> Callable[[str], int]:
    """Build a token-length function from a Hub tokenizer (downloads tokenizer.json)."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained(repo_id, token=token)
    return lambda text: len(tokenizer.encode(text).ids)
