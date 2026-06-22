"""Convert a finetuned run to a Core ML `.mlpackage`.

Supports the traceable engines: from-scratch PyTorch runs (a saved `model.pt`)
and HF-Trainer vision runs (a saved HF model directory). LLM/MLX runs aren't
traceable this way and raise a clear error.
"""

from __future__ import annotations

import os


def _first_linear_in_features(model) -> int:
    import torch.nn as nn

    for module in model.modules():
        if isinstance(module, nn.Linear):
            return module.in_features
    raise ValueError("no Linear layer found to infer the input shape")


def convert_run_to_coreml(run_path: str, out_path: str) -> str:
    import coremltools as ct
    import torch

    target = ct.target.macOS13

    pt = os.path.join(run_path, "model.pt")
    if os.path.exists(pt):
        model = torch.load(pt, weights_only=False).to("cpu").eval()
        example = torch.rand(1, _first_linear_in_features(model))
        traced = torch.jit.trace(model, example)
        mlmodel = ct.convert(
            traced, inputs=[ct.TensorType(name="input", shape=example.shape)],
            minimum_deployment_target=target,
        )
        mlmodel.save(out_path)
        return out_path

    if os.path.exists(os.path.join(run_path, "config.json")):
        from transformers import AutoModelForImageClassification

        model = AutoModelForImageClassification.from_pretrained(run_path).to("cpu").eval()
        size = int(getattr(model.config, "image_size", 224))

        class _Logits(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m(pixel_values=x).logits

        example = torch.rand(1, 3, size, size)
        traced = torch.jit.trace(_Logits(model), example)
        mlmodel = ct.convert(
            traced, inputs=[ct.TensorType(name="image", shape=example.shape)],
            minimum_deployment_target=target,
        )
        mlmodel.save(out_path)
        return out_path

    raise ValueError(
        "Core ML export supports from-scratch (PyTorch) and vision runs; "
        "this run has no traceable model."
    )
