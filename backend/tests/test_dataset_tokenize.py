"""Tests for tokenization rendering + length statistics."""

import pytest

from tinyforge.datasets.tokenize import compute_token_stats, render_for_tokenization


def test_render_text_format() -> None:
    assert render_for_tokenization({"text": "hello world"}, "text") == "hello world"


def test_render_completion_concatenates_prompt_and_completion() -> None:
    rendered = render_for_tokenization({"prompt": "Q ", "completion": "A"}, "completion")
    assert rendered == "Q A"


def test_render_chat_includes_all_message_contents() -> None:
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    rendered = render_for_tokenization({"messages": messages}, "chat")
    assert "hi" in rendered
    assert "yo" in rendered


def test_compute_token_stats_reports_distribution() -> None:
    texts = ["a", "a a", "a a a", "a a a a a"]  # word lengths 1, 2, 3, 5
    stats = compute_token_stats(texts, length_fn=lambda t: len(t.split()), bins=2)

    assert stats.count == 4
    assert stats.min == 1
    assert stats.max == 5
    assert stats.mean == pytest.approx(2.75)
    assert stats.p50 == 3  # nearest-rank median of [1,2,3,5]
    assert stats.p95 == 5
    assert len(stats.histogram) == 2
    assert sum(b.count for b in stats.histogram) == 4


def test_compute_token_stats_handles_empty() -> None:
    stats = compute_token_stats([], length_fn=lambda t: 0)
    assert stats.count == 0
    assert stats.histogram == []
