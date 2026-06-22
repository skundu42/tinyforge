"""Tests for parsing mlx_lm.lora training output into structured events."""

from tinyforge.train.parser import parse_line


def test_parse_train_metric_line() -> None:
    line = "Iter 1: Train loss 5.443, Learning Rate 1.000e-05, It/sec 0.554, Tokens/sec 6.651, Trained Tokens 12, Peak mem 0.182 GB"
    event = parse_line(line)
    assert event == {
        "event": "train",
        "iter": 1,
        "train_loss": 5.443,
        "lr": 1.0e-05,
        "it_per_sec": 0.554,
        "tokens_per_sec": 6.651,
        "trained_tokens": 12,
        "peak_mem_gb": 0.182,
    }


def test_parse_val_line() -> None:
    event = parse_line("Iter 2: Val loss 6.186, Val took 0.013s")
    assert event == {"event": "val", "iter": 2, "val_loss": 6.186}


def test_parse_saved_line() -> None:
    event = parse_line("Saved final weights to /tmp/tf-adapters/adapters.safetensors.")
    assert event == {"event": "saved", "path": "/tmp/tf-adapters/adapters.safetensors"}


def test_parse_trainable_params_line() -> None:
    event = parse_line("Trainable parameters: 0.242% (0.326M/134.515M)")
    assert event is not None
    assert event["event"] == "info"


def test_parse_ignores_progress_and_noise() -> None:
    assert parse_line("Calculating loss...: 100%|####| 1/1 [00:00<00:00, 86.05it/s]") is None
    assert parse_line("Loading datasets") is None
    assert parse_line("") is None
