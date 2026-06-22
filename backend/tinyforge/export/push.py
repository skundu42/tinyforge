"""Push an exported model folder to the HuggingFace Hub with a model card."""

from __future__ import annotations

import os


def push_folder(local_path: str, repo_id: str, base_model: str, token: str | None = None) -> str:
    from huggingface_hub import HfApi, ModelCard, ModelCardData

    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True, private=True)

    card = ModelCard.from_template(
        ModelCardData(
            base_model=base_model,
            tags=["tinyforge", "mlx", "lora", "finetuned"],
            library_name="mlx",
        ),
        model_description=(
            f"Finetuned from `{base_model}` with [TinyForge](https://github.com/) "
            "using MLX LoRA."
        ),
    )
    card.save(os.path.join(local_path, "README.md"))

    api.upload_folder(folder_path=local_path, repo_id=repo_id)
    return f"https://huggingface.co/{repo_id}"
