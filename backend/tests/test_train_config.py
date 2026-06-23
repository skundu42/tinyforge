"""Tests for building the mlx_lm.lora command from a RunConfig."""

from tinyforge.train.config import build_command
from tinyforge.train.models import RunConfig


def base_config(**kw) -> RunConfig:
    defaults = dict(
        name="run", model_repo="mlx-community/SmolLM-135M-Instruct-4bit",
        data_dir="/data/ds1", adapter_path="/runs/r1",
    )
    defaults.update(kw)
    return RunConfig(**defaults)


def test_build_command_uses_mlx_lm_lora_subcommand() -> None:
    command = build_command(base_config(), python_exe="/venv/bin/python3")
    assert command[:5] == ["/venv/bin/python3", "-m", "mlx_lm", "lora", "--train"]


def test_build_command_includes_core_flags() -> None:
    command = build_command(
        base_config(iters=200, batch_size=2, num_layers=8, learning_rate=2e-5), "py"
    )
    joined = " ".join(command)
    assert "--model mlx-community/SmolLM-135M-Instruct-4bit" in joined
    assert "--data /data/ds1" in joined
    assert "--adapter-path /runs/r1" in joined
    assert "--fine-tune-type lora" in joined
    assert "--iters 200" in joined
    assert "--batch-size 2" in joined
    assert "--num-layers 8" in joined
    assert "--learning-rate 2e-05" in joined


def test_grad_checkpoint_flag_toggles() -> None:
    assert "--grad-checkpoint" in build_command(base_config(grad_checkpoint=True), "py")
    assert "--grad-checkpoint" not in build_command(base_config(grad_checkpoint=False), "py")


def test_build_command_lm_engine_uses_lm_worker() -> None:
    cmd = build_command(
        base_config(
            engine="lm", iters=50, learning_rate=1e-3, batch_size=8,
            num_layers=4, hidden_size=128, num_heads=4, vocab_size=500, context_length=256,
        ),
        "py",
    )
    assert cmd[:3] == ["py", "-m", "tinyforge.train.lm_worker"]
    joined = " ".join(cmd)
    assert "--data /data/ds1" in joined
    assert "--adapter-path /runs/r1" in joined
    assert "--iters 50" in joined
    assert "--batch-size 8" in joined
    assert "--hidden-size 128" in joined
    assert "--num-layers 4" in joined
    assert "--num-heads 4" in joined
    assert "--vocab-size 500" in joined
    assert "--context-length 256" in joined
    assert "mlx_lm" not in joined
