"""From-scratch PyTorch/MPS training worker.

Trains a small MLP classifier on a synthetic non-linear task on the MPS device
(falling back to CPU), printing progress in the same format the mlx output
parser understands — so it reuses the run registry, WebSocket stream, and live
dashboards. This is the representative "from-scratch experimentation" path;
HF Trainer (vision/audio) and TRL workers follow the same event contract.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _synthetic_batch(n: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(n, 8, device=device)
    # A learnable non-linear decision boundary.
    label = (x[:, 0] * x[:, 1] + x[:, 2] - x[:, 3].abs() > 0).long()
    return x, label


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tinyforge.train.torch_worker")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = _device()
    model = nn.Sequential(
        nn.Linear(8, args.hidden), nn.ReLU(),
        nn.Linear(args.hidden, args.hidden), nn.ReLU(),
        nn.Linear(args.hidden, 2),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: 100.000% ({params / 1e6:.3f}M/{params / 1e6:.3f}M) on {device}", flush=True)
    print(f"Starting training..., iters: {args.iters}", flush=True)

    start = time.time()
    for it in range(1, args.iters + 1):
        x, y = _synthetic_batch(args.batch_size, device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

        if it == 1 or it % args.steps_per_report == 0:
            if device == "mps":
                torch.mps.synchronize()
            elapsed = max(time.time() - start, 1e-6)
            its = it / elapsed
            mem = torch.mps.current_allocated_memory() / 1e9 if device == "mps" else 0.0
            print(
                f"Iter {it}: Train loss {loss.item():.3f}, "
                f"Learning Rate {args.learning_rate:.3e}, It/sec {its:.3f}, "
                f"Tokens/sec {its * args.batch_size:.3f}, "
                f"Trained Tokens {it * args.batch_size}, Peak mem {mem:.3f} GB",
                flush=True,
            )
        if it % args.steps_per_eval == 0:
            with torch.no_grad():
                vx, vy = _synthetic_batch(256, device)
                vloss = loss_fn(model(vx), vy)
            print(f"Iter {it}: Val loss {vloss.item():.3f}, Val took 0.001s", flush=True)

    os.makedirs(args.adapter_path, exist_ok=True)
    out_path = os.path.join(args.adapter_path, "model.pt")
    # Save the whole module (on CPU) so it can be traced for Core ML export.
    torch.save(model.to("cpu"), out_path)
    print(f"Saved final weights to {out_path}.", flush=True)


if __name__ == "__main__":
    main()
