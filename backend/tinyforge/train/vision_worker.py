"""HuggingFace Trainer (vision) training worker.

Finetunes a small ViT image classifier on the MPS GPU using `transformers.Trainer`
on a synthetic, learnable image task, printing progress in the same format the
mlx output parser understands — so it reuses the run registry, WebSocket stream,
and live dashboards. This is the representative HF-Trainer path; audio
(Whisper / wav2vec) and real image datasets plug into the same worker + event
contract.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from torch.utils.data import Dataset


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class _SyntheticImages(Dataset):
    """Random 32x32 RGB images; label = whether the mean pixel is positive."""

    def __init__(self, n: int, seed: int) -> None:
        torch.manual_seed(seed)
        self.x = torch.randn(n, 3, 32, 32)
        self.y = (self.x.mean(dim=(1, 2, 3)) > 0).long()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> dict:
        return {"pixel_values": self.x[i], "labels": self.y[i]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tinyforge.train.vision_worker")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    import transformers
    from transformers import (
        Trainer,
        TrainerCallback,
        TrainingArguments,
        ViTConfig,
        ViTForImageClassification,
    )

    transformers.logging.set_verbosity_error()
    device = _device()

    config = ViTConfig(
        image_size=32, patch_size=8, num_channels=3, hidden_size=64,
        num_hidden_layers=2, num_attention_heads=2, intermediate_size=128, num_labels=2,
    )
    model = ViTForImageClassification(config)
    params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: 100.000% ({params / 1e6:.3f}M/{params / 1e6:.3f}M) "
        f"on {device} (ViT image classifier via HF Trainer)",
        flush=True,
    )
    print(f"Starting training..., iters: {args.iters}", flush=True)

    start = time.time()

    class _EmitEvents(TrainerCallback):
        def on_log(self, _args, state, _control, logs=None, **_kw):
            if not logs or "loss" not in logs:
                return
            elapsed = max(time.time() - start, 1e-6)
            its = state.global_step / elapsed
            mem = torch.mps.current_allocated_memory() / 1e9 if device == "mps" else 0.0
            seen = state.global_step * args.batch_size
            print(
                f"Iter {state.global_step}: Train loss {float(logs['loss']):.3f}, "
                f"Learning Rate {logs.get('learning_rate', args.learning_rate):.3e}, "
                f"It/sec {its:.3f}, Tokens/sec {its * args.batch_size:.3f}, "
                f"Trained Tokens {seen}, Peak mem {mem:.3f} GB",
                flush=True,
            )

    training_args = TrainingArguments(
        output_dir=args.adapter_path,
        max_steps=args.iters,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.steps_per_report,
        report_to=[],
        disable_tqdm=True,
        save_strategy="no",
        logging_strategy="steps",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=_SyntheticImages(max(256, args.batch_size * 8), args.seed),
        callbacks=[_EmitEvents()],
    )
    trainer.train()

    os.makedirs(args.adapter_path, exist_ok=True)
    trainer.save_model(args.adapter_path)
    print(f"Saved final weights to {os.path.join(args.adapter_path, 'model.safetensors')}.", flush=True)


if __name__ == "__main__":
    main()
