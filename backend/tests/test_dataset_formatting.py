"""Tests for dataset formatting into mlx-lm formats."""

import pytest

from tinyforge.datasets.formatting import format_row, target_format
from tinyforge.datasets.models import FormatSpec


def test_text_mode_passes_through_text_column() -> None:
    spec = FormatSpec(mode="text", text_column="content")
    assert format_row({"content": "hello"}, spec) == {"text": "hello"}
    assert target_format(spec) == "text"


def test_prompt_completion_mode() -> None:
    spec = FormatSpec(mode="prompt_completion", prompt_column="q", completion_column="a")
    assert format_row({"q": "Q?", "a": "A."}, spec) == {"prompt": "Q?", "completion": "A."}
    assert target_format(spec) == "completion"


def test_alpaca_mode_with_input_builds_prompt_and_completion() -> None:
    spec = FormatSpec(mode="alpaca")
    row = {"instruction": "Translate", "input": "hola", "output": "hello"}
    result = format_row(row, spec)
    assert result["prompt"] == "Translate\n\nhola"
    assert result["completion"] == "hello"
    assert target_format(spec) == "completion"


def test_alpaca_mode_without_input_omits_separator() -> None:
    spec = FormatSpec(mode="alpaca")
    row = {"instruction": "Say hi", "input": "", "output": "hi"}
    assert format_row(row, spec)["prompt"] == "Say hi"


def test_messages_mode_passes_through_list() -> None:
    spec = FormatSpec(mode="messages", messages_column="conversation")
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    assert format_row({"conversation": messages}, spec) == {"messages": messages}
    assert target_format(spec) == "chat"


def test_missing_column_raises_keyerror() -> None:
    spec = FormatSpec(mode="text", text_column="missing")
    with pytest.raises(KeyError):
        format_row({"other": "x"}, spec)
