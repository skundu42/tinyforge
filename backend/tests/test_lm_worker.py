import json
from pathlib import Path

import pytest

from tinyforge.train.lm_worker import build_llama_config

_E2E = Path(__file__).resolve().parents[2] / ".run-network-tests"


def test_build_llama_config_maps_knobs() -> None:
    cfg = build_llama_config(
        vocab_size=512, hidden_size=128, num_layers=4, num_heads=4, context_length=256
    )
    assert cfg.vocab_size == 512
    assert cfg.hidden_size == 128
    assert cfg.intermediate_size == 128 * 4
    assert cfg.num_hidden_layers == 4
    assert cfg.num_attention_heads == 4
    assert cfg.num_key_value_heads == 4
    assert cfg.max_position_embeddings == 256
    assert cfg.tie_word_embeddings is True


@pytest.mark.skipif(not _E2E.exists(), reason="opt-in: touch .run-network-tests")
def test_lm_worker_trains_and_saves_loadable_model(tmp_path, capsys) -> None:
    from tinyforge.train import lm_worker

    data = tmp_path / "ds"
    data.mkdir()
    line = json.dumps({"text": "the quick brown fox jumps over the lazy dog. " * 8})
    (data / "train.jsonl").write_text("\n".join([line] * 50) + "\n")
    (data / "valid.jsonl").write_text(line + "\n")
    out = tmp_path / "run"

    lm_worker.main([
        "--adapter-path", str(out), "--data", str(data), "--iters", "5",
        "--batch-size", "2", "--vocab-size", "300", "--context-length", "32",
        "--hidden-size", "64", "--num-layers", "2", "--num-heads", "2", "--steps-per-report", "1",
    ])

    printed = capsys.readouterr().out
    assert "Train loss" in printed
    assert (out / "model.safetensors").exists()
    assert (out / "config.json").exists()
    assert (out / "tokenizer.json").exists()
