"""From-scratch tiny-LM training worker.

Trains a randomly-initialized small Llama-style causal LM on the user's real text
dataset via the HuggingFace Trainer, printing progress in the exact format
`train/parser.py` understands so it reuses the run registry, WebSocket stream, and
live dashboards. Saves a standard HF checkpoint dir (config.json + model.safetensors
+ tokenizer) so the Playground and Export load it directly.
"""

from __future__ import annotations

import argparse
import os
import time

from tinyforge.train.lm_data import PackedTextDataset, load_corpus, pack_tokens
from tinyforge.train.tokenizer import train_bpe


def _device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def build_llama_config(
    vocab_size: int, hidden_size: int, num_layers: int, num_heads: int, context_length: int
):
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        max_position_embeddings=context_length,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
    )


def _pack_dataset(texts: list[str], tokenizer, block_size: int) -> PackedTextDataset:
    token_lists = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]
    blocks = pack_tokens(token_lists, block_size=block_size, eos_id=tokenizer.eos_token_id)
    return PackedTextDataset(blocks)


def main(argv: list[str] | None = None) -> None:
    import torch
    import transformers
    from transformers import (
        LlamaForCausalLM,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    parser = argparse.ArgumentParser(prog="tinyforge.train.lm_worker")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--context-length", type=int, default=512)
    args = parser.parse_args(argv)

    transformers.logging.set_verbosity_error()
    transformers.set_seed(args.seed)
    device = _device()

    train_texts, val_texts = load_corpus(args.data)
    if not train_texts:
        raise SystemExit(f"No training text found in {args.data}/train.jsonl")

    tokenizer = train_bpe(train_texts, vocab_size=args.vocab_size)
    train_ds = _pack_dataset(train_texts, tokenizer, args.context_length)
    val_ds = _pack_dataset(val_texts, tokenizer, args.context_length) if val_texts else None
    if len(train_ds) == 0:
        raise SystemExit(
            f"Corpus too small to fill one {args.context_length}-token block; add more data."
        )

    config = build_llama_config(
        vocab_size=len(tokenizer), hidden_size=args.hidden_size, num_layers=args.num_layers,
        num_heads=args.num_heads, context_length=args.context_length,
    )
    model = LlamaForCausalLM(config).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: 100.000% ({params / 1e6:.3f}M/{params / 1e6:.3f}M) "
        f"on {device} (Llama LM from scratch)",
        flush=True,
    )
    print(f"Starting training..., iters: {args.iters}", flush=True)

    start = time.time()

    class _EmitEvents(TrainerCallback):
        def on_log(self, _args, state, _control, logs=None, **_kw):
            if not logs:
                return
            elapsed = max(time.time() - start, 1e-6)
            its = state.global_step / elapsed
            mem = torch.mps.current_allocated_memory() / 1e9 if device == "mps" else 0.0
            seen = state.global_step * args.batch_size * args.context_length
            if "loss" in logs:
                print(
                    f"Iter {state.global_step}: Train loss {float(logs['loss']):.3f}, "
                    f"Learning Rate {logs.get('learning_rate', args.learning_rate):.3e}, "
                    f"It/sec {its:.3f}, Tokens/sec {its * args.batch_size * args.context_length:.3f}, "
                    f"Trained Tokens {seen}, Peak mem {mem:.3f} GB",
                    flush=True,
                )
            elif "eval_loss" in logs:
                print(f"Iter {state.global_step}: Val loss {float(logs['eval_loss']):.3f}", flush=True)

    training_args = TrainingArguments(
        output_dir=args.adapter_path,
        max_steps=args.iters,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.steps_per_report,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=args.steps_per_eval,
        report_to=[],
        disable_tqdm=True,
        save_strategy="no",
        logging_strategy="steps",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[_EmitEvents()],
    )
    trainer.train()

    os.makedirs(args.adapter_path, exist_ok=True)
    trainer.save_model(args.adapter_path)
    tokenizer.save_pretrained(args.adapter_path)
    print(f"Saved final weights to {os.path.join(args.adapter_path, 'model.safetensors')}.", flush=True)


if __name__ == "__main__":
    main()
