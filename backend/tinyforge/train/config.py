"""Build the mlx_lm.lora command line from a RunConfig."""

from __future__ import annotations

from tinyforge.train.models import RunConfig


def build_command(config: RunConfig, python_exe: str) -> list[str]:
    if config.engine == "lm":
        return [
            python_exe, "-m", "tinyforge.train.lm_worker",
            "--adapter-path", config.adapter_path,
            "--data", config.data_dir,
            "--iters", str(config.iters),
            "--learning-rate", str(config.learning_rate),
            "--batch-size", str(config.batch_size),
            "--steps-per-report", str(config.steps_per_report),
            "--steps-per-eval", str(config.steps_per_eval),
            "--seed", str(config.seed),
            "--hidden-size", str(config.hidden_size),
            "--num-layers", str(config.num_layers),
            "--num-heads", str(config.num_heads),
            "--vocab-size", str(config.vocab_size),
            "--context-length", str(config.context_length),
        ]

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
