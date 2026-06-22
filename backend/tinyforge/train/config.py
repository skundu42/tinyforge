"""Build the mlx_lm.lora command line from a RunConfig."""

from __future__ import annotations

from tinyforge.train.models import RunConfig


def build_command(config: RunConfig, python_exe: str) -> list[str]:
    command = [
        python_exe, "-m", "mlx_lm", "lora", "--train",
        "--model", config.model_repo,
        "--data", config.data_dir,
        "--adapter-path", config.adapter_path,
        "--fine-tune-type", config.fine_tune_type,
        "--num-layers", str(config.num_layers),
        "--batch-size", str(config.batch_size),
        "--iters", str(config.iters),
        "--learning-rate", str(config.learning_rate),
        "--steps-per-report", str(config.steps_per_report),
        "--steps-per-eval", str(config.steps_per_eval),
        "--val-batches", str(config.val_batches),
        "--max-seq-length", str(config.max_seq_length),
        "--seed", str(config.seed),
    ]
    if config.grad_checkpoint:
        command.append("--grad-checkpoint")
    return command
