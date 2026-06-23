# Tiny LM From-Scratch Training Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the synthetic toy training engines with a real `lm` engine that trains a small Llama-style causal LM from scratch on the user's text dataset and round-trips through the Playground and Export.

**Architecture:** A new `lm` engine runs an HF `Trainer` over a randomly-initialized `LlamaForCausalLM` on a corpus-trained BPE tokenizer, printing the same event lines the existing parser/dashboards already consume. Output is a standard HF checkpoint dir so `mlx_lm.load()` (Playground) and `mlx_lm convert` (Export) work directly. The `torch` and `vision` toy engines are removed.

**Tech Stack:** Python 3.12, PyTorch/MPS, HuggingFace `transformers` + `tokenizers` + `datasets`, FastAPI, `mlx-lm`; SwiftUI (Swift Testing framework), `uv`, `pytest`.

## Global Constraints

- Python `requires-python = ">=3.12"`; line-length 100 (ruff default lint set, E501 not enforced).
- Backend tests: `cd backend && uv run pytest`. App tests: `cd App && xcodebuild test -scheme TinyForge -destination 'platform=macOS' -skipMacroValidation`.
- Swift tests use the **Swift Testing** framework (`import Testing`, `@Test`, `#expect`) — NOT XCTest.
- Event output lines MUST match `backend/tinyforge/train/parser.py` regexes exactly (train/val/saved/params).
- Engine values after this work: `Literal["mlx", "lm"]` only. No `torch`, no `vision`.
- Export targets after this work: `Literal["safetensors", "mlx"]` only. No `gguf`, no `coreml`.
- Heavy/network integration tests are opt-in: gated behind `Path(".run-network-tests").exists()` (repo convention).
- Commit after every task. Branch is `feat/tiny-lm-engine` (already created).

---

### Task 1: LM knobs + presets in train models

**Files:**
- Modify: `backend/tinyforge/train/models.py`
- Test: `backend/tests/test_train_models.py` (create)

**Interfaces:**
- Produces: `RunConfig`/`StartRunRequest` gain `model_size: str`, `hidden_size: int`, `num_heads: int`, `vocab_size: int`, `context_length: int`; `engine` Literal narrows to `["mlx", "lm"]`. New `LM_PRESETS: dict[str, dict]` and `apply_preset(model_size, num_layers, hidden_size, num_heads, context_length) -> tuple[int,int,int,int]` returning `(num_layers, hidden_size, num_heads, context_length)`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_train_models.py
from tinyforge.train.models import LM_PRESETS, RunConfig, StartRunRequest, apply_preset


def test_engine_literal_allows_mlx_and_lm() -> None:
    assert RunConfig(name="r", model_repo="m", data_dir="/d", adapter_path="/a", engine="lm").engine == "lm"
    assert StartRunRequest(name="r", engine="lm").engine == "lm"


def test_lm_request_has_model_knobs_with_defaults() -> None:
    req = StartRunRequest(name="r", engine="lm")
    assert req.model_size == "small"
    assert req.hidden_size == 256
    assert req.num_heads == 8
    assert req.vocab_size == 8000
    assert req.context_length == 512


def test_apply_preset_expands_named_sizes() -> None:
    # tiny preset overrides whatever was passed
    assert apply_preset("tiny", 16, 999, 999, 999) == (4, 128, 4, 256)
    assert apply_preset("small", 16, 999, 999, 999) == (6, 256, 8, 512)
    assert apply_preset("medium", 16, 999, 999, 999) == (8, 512, 8, 512)


def test_apply_preset_custom_passes_values_through() -> None:
    assert apply_preset("custom", 5, 384, 6, 1024) == (5, 384, 6, 1024)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_train_models.py -q`
Expected: FAIL with `ImportError: cannot import name 'LM_PRESETS'`.

- [ ] **Step 3: Write minimal implementation**

In `backend/tinyforge/train/models.py`, change the engine Literal and add LM fields + presets. Replace the `engine` line in BOTH `RunConfig` and `StartRunRequest` with `engine: Literal["mlx", "lm"] = "mlx"` (StartRunRequest currently uses `str`; make it the Literal too), and add the new fields after `seed`:

```python
# In RunConfig (after seed: int = 0):
    model_size: str = "small"      # tiny | small | medium | custom (lm engine)
    hidden_size: int = 256
    num_heads: int = 8
    vocab_size: int = 8000
    context_length: int = 512

# In StartRunRequest (after seed: int = 0):
    model_size: str = "small"
    hidden_size: int = 256
    num_heads: int = 8
    vocab_size: int = 8000
    context_length: int = 512
```

Change `StartRunRequest.engine: str = "mlx"` to `engine: Literal["mlx", "lm"] = "mlx"` and `RunConfig.engine: Literal["mlx", "torch", "vision"] = "mlx"` to `engine: Literal["mlx", "lm"] = "mlx"`.

Add at module scope (below the imports):

```python
# (num_layers, hidden_size, num_heads, context_length)
LM_PRESETS: dict[str, tuple[int, int, int, int]] = {
    "tiny": (4, 128, 4, 256),
    "small": (6, 256, 8, 512),
    "medium": (8, 512, 8, 512),
}


def apply_preset(
    model_size: str, num_layers: int, hidden_size: int, num_heads: int, context_length: int
) -> tuple[int, int, int, int]:
    """Resolve LM dimensions: named presets win; 'custom' (or unknown) passes values through."""
    if model_size in LM_PRESETS:
        return LM_PRESETS[model_size]
    return (num_layers, hidden_size, num_heads, context_length)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_train_models.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/tinyforge/train/models.py backend/tests/test_train_models.py
git commit -m "feat(train): add lm engine knobs + model-size presets"
```

---

### Task 2: Corpus → text rendering + token packing (`lm_data`)

**Files:**
- Create: `backend/tinyforge/train/lm_data.py`
- Test: `backend/tests/test_lm_data.py` (create)

**Interfaces:**
- Produces:
  - `render_text(row: dict) -> str`
  - `pack_tokens(token_lists: list[list[int]], block_size: int, eos_id: int) -> list[list[int]]`
  - `load_corpus(data_dir: str) -> tuple[list[str], list[str]]` (train_texts, val_texts) reading `train.jsonl`/`valid.jsonl`
  - `PackedTextDataset(blocks: list[list[int]])` — a `torch.utils.data.Dataset` yielding `{"input_ids": LongTensor, "labels": LongTensor}`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_lm_data.py
import json

from tinyforge.train.lm_data import load_corpus, pack_tokens, render_text


def test_render_text_handles_each_format() -> None:
    assert render_text({"text": "hello"}) == "hello"
    assert render_text({"prompt": "Q?", "completion": "A."}) == "Q?\n\nA."
    assert render_text(
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    ) == "user: hi\nassistant: yo"


def test_pack_tokens_concatenates_with_eos_and_chunks() -> None:
    # two docs [1,2] and [3]; eos=0; block=3 -> [1,2,0],[3,0] (last short chunk dropped)
    blocks = pack_tokens([[1, 2], [3]], block_size=3, eos_id=0)
    assert blocks == [[1, 2, 0]]


def test_pack_tokens_keeps_all_full_blocks() -> None:
    blocks = pack_tokens([[1, 1, 1], [2, 2]], block_size=2, eos_id=9)
    # stream: 1,1,1,9,2,2,9 -> blocks of 2: [1,1],[1,9],[2,2]; trailing [9] dropped
    assert blocks == [[1, 1], [1, 9], [2, 2]]


def test_load_corpus_reads_jsonl(tmp_path) -> None:
    (tmp_path / "train.jsonl").write_text(json.dumps({"text": "a"}) + "\n" + json.dumps({"text": "b"}) + "\n")
    (tmp_path / "valid.jsonl").write_text(json.dumps({"text": "c"}) + "\n")
    train, val = load_corpus(str(tmp_path))
    assert train == ["a", "b"]
    assert val == ["c"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_lm_data.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'tinyforge.train.lm_data'`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/tinyforge/train/lm_data.py
"""Turn a prepared text dataset into packed token blocks for from-scratch LM training.

Reads the `train.jsonl` / `valid.jsonl` the dataset builder writes (text / prompt-
completion / messages rows), renders each row to plain text, tokenizes, concatenates
documents with an EOS separator, and chunks into fixed-length blocks for causal LM.
"""

from __future__ import annotations

import json
import os
from typing import Any


def render_text(row: dict[str, Any]) -> str:
    """Render one prepared dataset row to a single training string."""
    if "text" in row:
        return str(row["text"])
    if "prompt" in row and "completion" in row:
        return f"{row['prompt']}\n\n{row['completion']}"
    if "messages" in row:
        return "\n".join(f"{m['role']}: {m['content']}" for m in row["messages"])
    raise ValueError(f"row has no recognized text fields: {sorted(row)}")


def load_corpus(data_dir: str) -> tuple[list[str], list[str]]:
    """Read train/valid jsonl from a prepared dataset dir into lists of rendered text."""
    def _read(name: str) -> list[str]:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as handle:
            return [render_text(json.loads(line)) for line in handle if line.strip()]

    return _read("train.jsonl"), _read("valid.jsonl")


def pack_tokens(token_lists: list[list[int]], block_size: int, eos_id: int) -> list[list[int]]:
    """Concatenate token lists (EOS between docs) and split into full `block_size` blocks."""
    stream: list[int] = []
    for ids in token_lists:
        stream.extend(ids)
        stream.append(eos_id)
    n_blocks = len(stream) // block_size
    return [stream[i * block_size : (i + 1) * block_size] for i in range(n_blocks)]


class PackedTextDataset:
    """A torch Dataset of fixed-length blocks; labels == input_ids for causal LM."""

    def __init__(self, blocks: list[list[int]]) -> None:
        self._blocks = blocks

    def __len__(self) -> int:
        return len(self._blocks)

    def __getitem__(self, i: int) -> dict:
        import torch

        ids = torch.tensor(self._blocks[i], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_lm_data.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/tinyforge/train/lm_data.py backend/tests/test_lm_data.py
git commit -m "feat(train): corpus text rendering + token packing for lm engine"
```

---

### Task 3: Corpus-trained BPE tokenizer (`tokenizer`)

**Files:**
- Create: `backend/tinyforge/train/tokenizer.py`
- Test: `backend/tests/test_lm_tokenizer.py` (create)

**Interfaces:**
- Produces: `train_bpe(texts: Iterable[str], vocab_size: int) -> PreTrainedTokenizerFast` with `eos_token = pad_token = "<|endoftext|>"`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_lm_tokenizer.py
from tinyforge.train.tokenizer import train_bpe


def test_train_bpe_roundtrips_text() -> None:
    corpus = ["hello world", "hello there", "world of words"] * 20
    tok = train_bpe(corpus, vocab_size=300)
    ids = tok("hello world")["input_ids"]
    assert len(ids) > 0
    assert "hello" in tok.decode(ids)


def test_train_bpe_sets_special_tokens() -> None:
    tok = train_bpe(["abc def ghi"] * 20, vocab_size=300)
    assert tok.eos_token == "<|endoftext|>"
    assert tok.pad_token == "<|endoftext|>"
    assert tok.eos_token_id is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_lm_tokenizer.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'tinyforge.train.tokenizer'`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/tinyforge/train/tokenizer.py
"""Train a small byte-level BPE tokenizer on the user's corpus (truly from scratch).

Produces a `PreTrainedTokenizerFast` that `save_pretrained`s into a standard
`tokenizer.json` + `tokenizer_config.json`, so the trained model round-trips in
`transformers` and `mlx_lm`.
"""

from __future__ import annotations

from collections.abc import Iterable

EOS = "<|endoftext|>"


def train_bpe(texts: Iterable[str], vocab_size: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(list(texts), trainer)

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token=EOS,
        pad_token=EOS,
        bos_token=EOS,
        unk_token=EOS,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_lm_tokenizer.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/tinyforge/train/tokenizer.py backend/tests/test_lm_tokenizer.py
git commit -m "feat(train): corpus-trained BPE tokenizer for lm engine"
```

---

### Task 4: LM worker — model config + training entry point

**Files:**
- Create: `backend/tinyforge/train/lm_worker.py`
- Test: `backend/tests/test_lm_worker.py` (create)

**Interfaces:**
- Consumes: `lm_data.load_corpus`, `lm_data.pack_tokens`, `lm_data.PackedTextDataset`, `tokenizer.train_bpe`.
- Produces:
  - `build_llama_config(vocab_size, hidden_size, num_layers, num_heads, context_length) -> LlamaConfig`
  - `main(argv: list[str] | None = None) -> None` CLI entry point (module runnable via `-m tinyforge.train.lm_worker`).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_lm_worker.py
from tinyforge.train.lm_worker import build_llama_config


def test_build_llama_config_maps_knobs() -> None:
    cfg = build_llama_config(
        vocab_size=512, hidden_size=128, num_layers=4, num_heads=4, context_length=256
    )
    assert cfg.vocab_size == 512
    assert cfg.hidden_size == 128
    assert cfg.num_hidden_layers == 4
    assert cfg.num_attention_heads == 4
    assert cfg.num_key_value_heads == 4
    assert cfg.max_position_embeddings == 256
    assert cfg.tie_word_embeddings is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_lm_worker.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'tinyforge.train.lm_worker'`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/tinyforge/train/lm_worker.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_lm_worker.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Add an opt-in end-to-end smoke test**

```python
# append to backend/tests/test_lm_worker.py
import json
from pathlib import Path

import pytest

_E2E = Path(__file__).resolve().parents[2] / ".run-network-tests"


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
```

- [ ] **Step 6: Run the fast test again (smoke stays skipped)**

Run: `cd backend && uv run pytest tests/test_lm_worker.py -q`
Expected: PASS with 1 passed, 1 skipped.

- [ ] **Step 7: Commit**

```bash
git add backend/tinyforge/train/lm_worker.py backend/tests/test_lm_worker.py
git commit -m "feat(train): from-scratch tiny-LM training worker (HF Trainer + Llama)"
```

---

### Task 5: Command builder — `lm` branch, drop torch/vision

**Files:**
- Modify: `backend/tinyforge/train/config.py`
- Modify: `backend/tests/test_train_config.py`

**Interfaces:**
- Consumes: `RunConfig` (Task 1) with LM knobs.
- Produces: `build_command(config, python_exe)` routes `engine == "lm"` to `tinyforge.train.lm_worker` with all LM args; `mlx` branch unchanged; torch/vision branches removed.

- [ ] **Step 1: Replace the torch/vision tests with an lm test**

In `backend/tests/test_train_config.py`, delete `test_build_command_torch_engine_uses_torch_worker` and `test_build_command_vision_engine_uses_vision_worker` (lines 41-58), and add:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_train_config.py -q`
Expected: FAIL — `lm` currently falls through to the `mlx_lm lora` branch, so `mlx_lm` IS in the command and the worker assertion fails.

- [ ] **Step 3: Rewrite the engine branch in `config.py`**

Replace the entire `if config.engine in ("torch", "vision"): ...` block (lines 9-29) with:

```python
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
```

(Leave the `mlx_lm lora` command block that follows unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_train_config.py -q`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add backend/tinyforge/train/config.py backend/tests/test_train_config.py
git commit -m "feat(train): route lm engine to lm_worker; drop torch/vision branches"
```

---

### Task 6: Service wiring + derived `engine` on RunRecord

**Files:**
- Modify: `backend/tinyforge/train/service.py`
- Modify: `backend/tinyforge/train/models.py`
- Modify: `backend/tinyforge/train/registry.py`
- Modify: `backend/tests/test_train_service.py`
- Test: `backend/tests/test_run_registry.py` (add a case)

**Interfaces:**
- Produces:
  - `RunRecord` gains `engine: str = "mlx"` (derived, not a DB column).
  - `TrainingService.start`: always resolves `data_dir` via the resolver; for `lm`, expands the preset and sets `model_repo = adapter_path`; persists `engine`.

- [ ] **Step 1: Write failing service tests**

In `backend/tests/test_train_service.py`, add:

```python
def test_lm_run_resolves_dataset_and_points_model_at_output_dir(tmp_path) -> None:
    svc, runner, registry = service(tmp_path)

    record = svc.start(request(engine="lm", model_size="tiny", dataset_id="dsX"))

    cfg = runner.started["config"]
    assert cfg.engine == "lm"
    assert cfg.data_dir == "/data/dsX"
    # from-scratch model lives in its own run dir, so model_repo == adapter_path
    assert cfg.model_repo == cfg.adapter_path
    # tiny preset applied
    assert (cfg.num_layers, cfg.hidden_size, cfg.num_heads, cfg.context_length) == (4, 128, 4, 256)
    assert record.engine == "lm"


def test_lm_custom_size_passes_knobs_through(tmp_path) -> None:
    svc, runner, _ = service(tmp_path)
    svc.start(request(
        engine="lm", model_size="custom", num_layers=3, hidden_size=192, num_heads=6,
        context_length=128,
    ))
    cfg = runner.started["config"]
    assert (cfg.num_layers, cfg.hidden_size, cfg.num_heads, cfg.context_length) == (3, 192, 6, 128)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_train_service.py -q`
Expected: FAIL — `record.engine` does not exist and preset/model_repo logic is absent.

- [ ] **Step 3: Add `engine` to RunRecord**

In `backend/tinyforge/train/models.py`, add to `RunRecord` (after `state: str`):

```python
    engine: str = "mlx"
```

- [ ] **Step 4: Make the registry derive `engine` without a schema change**

In `backend/tinyforge/train/registry.py`:

In `save`, change the model_dump exclusion to also drop the derived field:

```python
                {**record.model_dump(exclude={"config", "engine"}), "config": json.dumps(record.config)},
```

In `_to_record`, derive `engine` from the stored config before constructing the record:

```python
    @staticmethod
    def _to_record(row: sqlite3.Row) -> RunRecord:
        data = dict(row)
        data["config"] = json.loads(data["config"])
        data["engine"] = data["config"].get("engine", "mlx")
        return RunRecord(**data)
```

- [ ] **Step 5: Update `TrainingService.start`**

In `backend/tinyforge/train/service.py`, replace lines 46-50 (the `data_dir` / `default_names` / `model_repo` / `adapter_path` block) with:

```python
        data_dir = self._resolve_dataset(request.dataset_id)
        adapter_path = str(self._runs_dir / run_id)
        # A from-scratch LM has no base repo; its model lives in the run's own dir.
        model_repo = adapter_path if request.engine == "lm" else request.model_repo
        num_layers, hidden_size, num_heads, context_length = apply_preset(
            request.model_size, request.num_layers, request.hidden_size,
            request.num_heads, request.context_length,
        )
```

Update the `RunConfig(...)` construction to pass the resolved LM knobs and engine fields. Replace the `num_layers=request.num_layers,` argument and add the new ones; the full `config = RunConfig(...)` becomes:

```python
        config = RunConfig(
            name=request.name, model_repo=model_repo, data_dir=data_dir,
            adapter_path=adapter_path, engine=request.engine,
            fine_tune_type=request.fine_tune_type,
            num_layers=num_layers, batch_size=request.batch_size,
            iters=request.iters, learning_rate=request.learning_rate,
            steps_per_report=request.steps_per_report, steps_per_eval=request.steps_per_eval,
            max_seq_length=request.max_seq_length, grad_checkpoint=request.grad_checkpoint,
            seed=request.seed, model_size=request.model_size, hidden_size=hidden_size,
            num_heads=num_heads, vocab_size=request.vocab_size, context_length=context_length,
        )
```

Add `engine=request.engine` to the `RunRecord(...)` construction (after `name=request.name`):

```python
        record = RunRecord(
            id=run_id, name=request.name, engine=request.engine, model_repo=model_repo,
            dataset_id=request.dataset_id, state="running", created_at=self._clock(),
            adapter_path=adapter_path, config=config.model_dump(),
        )
```

Add the import at the top of `service.py`:

```python
from tinyforge.train.models import RunConfig, RunRecord, RunStatus, StartRunRequest, apply_preset
```

- [ ] **Step 6: Add a registry test for the derived field**

In `backend/tests/test_run_registry.py`, add:

```python
def test_engine_is_derived_from_config(tmp_path) -> None:
    from tinyforge.train.models import RunConfig, RunRecord
    from tinyforge.train.registry import RunRegistry

    reg = RunRegistry(tmp_path)
    cfg = RunConfig(name="r", model_repo="/runs/r", data_dir="/d", adapter_path="/runs/r", engine="lm")
    reg.save(RunRecord(
        id="r", name="r", engine="lm", model_repo="/runs/r", dataset_id="d",
        state="completed", created_at="t", adapter_path="/runs/r", config=cfg.model_dump(),
    ))
    assert reg.get("r").engine == "lm"
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_train_service.py tests/test_run_registry.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add backend/tinyforge/train/service.py backend/tinyforge/train/models.py backend/tinyforge/train/registry.py backend/tests/test_train_service.py backend/tests/test_run_registry.py
git commit -m "feat(train): lm service wiring (presets, output-dir model) + derived engine on RunRecord"
```

---

### Task 7: Export rework — full-model branch, narrow targets

**Files:**
- Modify: `backend/tinyforge/export/models.py`
- Modify: `backend/tinyforge/export/manager.py`
- Modify: `backend/tinyforge/services.py`
- Modify: `backend/tests/test_export_manager.py`

**Interfaces:**
- Consumes: `build_convert_command` (unchanged), the run's `engine` (from `record.config["engine"]`).
- Produces:
  - `ExportRequest.target: Literal["safetensors", "mlx"]`.
  - `RunResolver = Callable[[str], tuple[str, str, str]]` returning `(model_repo, adapter_path, engine)`.
  - `ExportManager` branches: `engine == "lm"` → no fuse (safetensors = copy run dir; mlx = `convert --hf-path run_dir`); `engine == "mlx"` → existing fuse(+convert) path. No `coreml_fn`.

- [ ] **Step 1: Rewrite the export-manager tests**

Replace the whole body of `backend/tests/test_export_manager.py` with:

```python
"""Tests for ExportManager: LoRA-adapter fuse/convert vs full-model (lm) export."""

import time

from tinyforge.export.manager import ExportManager
from tinyforge.export.models import ExportRequest


def _wait(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def manager(tmp_path, *, run_command, engine="mlx", push_fn=None):
    return ExportManager(
        python_exe="py", exports_dir=tmp_path,
        run_resolver=lambda run_id: (f"/runs/{run_id}" if engine == "lm" else "base/m",
                                     f"/runs/{run_id}", engine),
        run_command=run_command,
        push_fn=push_fn or (lambda path, repo, base: f"https://hf.co/{repo}"),
        id_factory=lambda: "exp1",
    )


# --- LoRA-adapter (mlx) runs: existing fuse-based path -----------------------

def test_adapter_safetensors_runs_only_fuse(tmp_path) -> None:
    log: list[list[str]] = []
    mgr = manager(tmp_path, run_command=lambda cmd, cwd: (log.append(cmd) or (0, "")))
    job_id = mgr.start(ExportRequest(run_id="r1", target="safetensors"))
    assert _wait(lambda: mgr.status(job_id).state == "completed")
    assert len(log) == 1 and "fuse" in log[0]
    assert mgr.status(job_id).output_path.endswith("fused")


def test_adapter_mlx_runs_fuse_then_convert(tmp_path) -> None:
    log: list[list[str]] = []
    mgr = manager(tmp_path, run_command=lambda cmd, cwd: (log.append(cmd) or (0, "")))
    job_id = mgr.start(ExportRequest(run_id="r1", target="mlx", q_bits=4))
    assert _wait(lambda: mgr.status(job_id).state == "completed")
    assert len(log) == 2 and "convert" in log[1]
    assert mgr.status(job_id).output_path.endswith("mlx")


# --- Full-model (lm) runs: no fuse -------------------------------------------

def test_lm_safetensors_copies_run_dir_no_fuse(tmp_path) -> None:
    (tmp_path).joinpath("dummy").write_text("x")  # ensure tmp exists
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "model.safetensors").write_text("weights")
    log: list[list[str]] = []
    mgr = manager(tmp_path, run_command=lambda cmd, cwd: (log.append(cmd) or (0, "")), engine="lm")
    job_id = mgr.start(ExportRequest(run_id="r1", target="safetensors"))
    assert _wait(lambda: mgr.status(job_id).state == "completed")
    assert log == []  # no fuse, no subprocess
    out = mgr.status(job_id).output_path
    assert out.endswith("model")


def test_lm_mlx_converts_run_dir_directly(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    log: list[list[str]] = []
    mgr = manager(tmp_path, run_command=lambda cmd, cwd: (log.append(cmd) or (0, "")), engine="lm")
    job_id = mgr.start(ExportRequest(run_id="r1", target="mlx", q_bits=4))
    assert _wait(lambda: mgr.status(job_id).state == "completed")
    assert len(log) == 1 and "convert" in log[0]
    joined = " ".join(log[0])
    assert "--hf-path /runs/r1" in joined  # converts the run dir, not a fused dir


def test_fuse_failure_marks_failed(tmp_path) -> None:
    mgr = manager(tmp_path, run_command=lambda cmd, cwd: (1, "boom error detail"))
    job_id = mgr.start(ExportRequest(run_id="r1", target="safetensors"))
    assert _wait(lambda: mgr.status(job_id).state == "failed")
    assert "boom" in mgr.status(job_id).error


def test_push_invoked_when_repo_set(tmp_path) -> None:
    pushes: list[tuple] = []
    mgr = manager(
        tmp_path, run_command=lambda cmd, cwd: (0, ""),
        push_fn=lambda path, repo, base: (pushes.append((repo, base)) or "https://hf.co/me/m"),
    )
    job_id = mgr.start(ExportRequest(run_id="r1", target="safetensors", push_repo="me/m"))
    assert _wait(lambda: mgr.status(job_id).state == "completed")
    assert pushes == [("me/m", "base/m")]
    assert mgr.status(job_id).hub_url == "https://hf.co/me/m"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_export_manager.py -q`
Expected: FAIL — resolver now returns a 3-tuple and the lm branch/target Literal don't exist.

- [ ] **Step 3: Narrow the target Literal**

In `backend/tinyforge/export/models.py`, change line 12 to:

```python
    target: Literal["safetensors", "mlx"] = "safetensors"
```

- [ ] **Step 4: Rework the manager**

In `backend/tinyforge/export/manager.py`:

Delete the `_default_coreml` function (lines 34-37), the `CoreMLFn` type (lines 40-41), the `from tinyforge.export.coreml ...` import inside it, the `coreml_fn` constructor parameter (line 63) and its assignment (`self._coreml_fn = coreml_fn`). Update the `RunResolver` type comment/alias to:

```python
# run_resolver(run_id) -> (model_repo, adapter_path, engine)
RunResolver = Callable[[str], tuple[str, str, str]]
```

Replace `_run` (lines 83-134) with:

```python
    def _run(self, job: _Export) -> None:
        import shutil

        request = job.request
        model_repo, adapter_path, engine = self._resolve_run(request.run_id)
        out_dir = self._exports_dir / job.id
        out_dir.mkdir(parents=True, exist_ok=True)

        if engine == "lm":
            # Full from-scratch model: the run dir IS the model — no fuse.
            if request.target == "safetensors":
                dest = out_dir / "model"
                shutil.copytree(model_repo, dest, dirs_exist_ok=True)
                result_path = str(dest)
            else:  # mlx
                mlx_path = out_dir / "mlx"
                code, output = self._run_command(
                    build_convert_command(self._python, model_repo, str(mlx_path), request.q_bits),
                    str(out_dir),
                )
                if code != 0:
                    return self._fail(job, output)
                result_path = str(mlx_path)
        else:
            # LoRA adapter on a base repo: fuse, then optionally convert.
            fused = out_dir / "fused"
            code, output = self._run_command(
                build_fuse_command(self._python, model_repo, adapter_path, str(fused)),
                str(out_dir),
            )
            if code != 0:
                return self._fail(job, output)
            result_path = str(fused)
            if request.target == "mlx":
                mlx_path = out_dir / "mlx"
                code, output = self._run_command(
                    build_convert_command(self._python, str(fused), str(mlx_path), request.q_bits),
                    str(out_dir),
                )
                if code != 0:
                    return self._fail(job, output)
                result_path = str(mlx_path)

        if request.push_repo:
            if self._push_fn is None:
                return self._fail(job, "push requested but no pusher configured")
            try:
                url = self._push_fn(result_path, request.push_repo, model_repo)
            except Exception as exc:  # noqa: BLE001
                return self._fail(job, f"push failed: {exc}")
            with self._lock:
                job.hub_url = url

        with self._lock:
            job.output_path = result_path
            job.state = "completed"
```

- [ ] **Step 5: Update the resolver in `services.py`**

In `backend/tinyforge/services.py`, change `resolve_run` (lines 53-55) to return the engine and drop the no-longer-passed `coreml_fn`:

```python
    def resolve_run(run_id: str) -> tuple[str, str, str]:
        record = training.get(run_id)
        return record.config["model_repo"], record.adapter_path, record.config.get("engine", "mlx")
```

(The `ExportManager(...)` call on lines 57-60 already omits `coreml_fn`; leave it.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_export_manager.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/tinyforge/export/models.py backend/tinyforge/export/manager.py backend/tinyforge/services.py backend/tests/test_export_manager.py
git commit -m "feat(export): full-model (lm) export path; drop gguf/coreml targets"
```

---

### Task 8: Remove demo engines + add explicit transformers dep

**Files:**
- Delete: `backend/tinyforge/train/torch_worker.py`, `backend/tinyforge/train/vision_worker.py`, `backend/tinyforge/export/coreml.py`
- Modify: `backend/pyproject.toml`
- Delete test references: `backend/tests/test_train_parser.py` and any others referencing torch/vision workers (verify with grep)

**Interfaces:** none (pure removal + dependency declaration).

- [ ] **Step 1: Find every reference to the demo engines**

Run: `cd backend && grep -rn "torch_worker\|vision_worker\|coreml\|_synthetic\|SyntheticImages\|from-scratch MLP\|ViT image classifier" tinyforge tests`
Expected: a list. Every hit must be removed or already handled by Tasks 5-7. Note any test files that exercise the deleted workers.

- [ ] **Step 2: Delete the worker + coreml files**

```bash
git rm backend/tinyforge/train/torch_worker.py backend/tinyforge/train/vision_worker.py backend/tinyforge/export/coreml.py
```

- [ ] **Step 3: Remove dead tests**

For any test file surfaced in Step 1 that targets the deleted workers (e.g. a synthetic-data test, a coreml test), delete the file or the specific tests. The parser tests in `test_train_parser.py` are format-based (not engine-specific) and should remain — only remove assertions that mention `torch`/`vision` engines if present.

- [ ] **Step 4: Add transformers as an explicit dependency**

In `backend/pyproject.toml`, add to `dependencies` (after `"accelerate>=1.14.0",`):

```toml
    "transformers>=4.44",
```

Remove `"coremltools>=9.0",` from `dependencies` (only the deleted Core ML export used it).

- [ ] **Step 5: Sync and run the full backend suite**

```bash
cd backend && uv sync && uv run pytest -q
```
Expected: PASS, no import errors, no references to deleted modules.

- [ ] **Step 6: Commit**

```bash
git add -A backend
git commit -m "chore(train): remove synthetic torch/vision demo engines + coreml export"
```

---

### Task 9: Swift — StartRunRequest + TrainingModel for the lm engine

**Files:**
- Modify: `App/Sources/Backend/TrainModels.swift`
- Modify: `App/Sources/Features/Training/TrainingModel.swift`
- Modify: `App/Tests/TrainingModelTests.swift`

**Interfaces:**
- Produces: `StartRunRequest` gains `modelSize`, `hiddenSize`, `numHeads`, `vocabSize`, `contextLength`. `TrainingModel` gains matching fields, `engine` defaulting to `mlx` with `lm` option, and `canStart` for `lm` requiring `datasetId` (not `modelRepo`).

- [ ] **Step 1: Update the failing Swift tests**

In `App/Tests/TrainingModelTests.swift`, replace `torchEngineCanStartWithoutModelOrDataset` (lines 23-29) with:

```swift
    @Test func lmEngineCanStartWithDatasetButNoModel() {
        let api = FakeBackendAPI()
        let sut = TrainingModel(api: api, events: FakeRunEventStreaming(events: []))
        sut.name = "scratch"
        sut.engine = "lm"
        sut.datasetId = "ds1"
        sut.modelRepo = ""        // from-scratch needs no base model…
        #expect(sut.canStart)
        sut.datasetId = ""        // …but it does need a dataset
        #expect(!sut.canStart)
    }

    @Test func lmStartSendsModelSizeAndKnobs() async {
        let api = FakeBackendAPI()
        let sut = TrainingModel(api: api, events: FakeRunEventStreaming(events: []))
        sut.name = "scratch"; sut.engine = "lm"; sut.datasetId = "ds1"; sut.modelSize = "tiny"
        await sut.start()
        #expect(api.startedRequest?.engine == "lm")
        #expect(api.startedRequest?.modelSize == "tiny")
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd App && xcodebuild test -scheme TinyForge -destination 'platform=macOS' -skipMacroValidation 2>&1 | tail -30`
Expected: FAIL — `modelSize` and `lm` handling don't exist.

- [ ] **Step 3: Extend `StartRunRequest`**

In `App/Sources/Backend/TrainModels.swift`, add fields after `seed` (line 17):

```swift
    var modelSize: String = "small"
    var hiddenSize: Int = 256
    var numHeads: Int = 8
    var vocabSize: Int = 8000
    var contextLength: Int = 512
```

Add their coding keys inside `enum CodingKeys` (after `case gradCheckpoint = "grad_checkpoint"`):

```swift
        case modelSize = "model_size"
        case hiddenSize = "hidden_size"
        case numHeads = "num_heads"
        case vocabSize = "vocab_size"
        case contextLength = "context_length"
```

- [ ] **Step 4: Update `TrainingModel`**

In `App/Sources/Features/Training/TrainingModel.swift`:

Change the engine comment/field (line 11) and add LM config fields after `maxSeqLength` (line 19):

```swift
    var engine = "mlx"  // mlx (LLM LoRA) | lm (from-scratch tiny LM)
    var modelSize = "small"
    var hiddenSize = 256
    var numHeads = 8
    var vocabSize = 8000
    var contextLength = 512
```

Replace `isLLM` and `canStart` (lines 50-54) with:

```swift
    var isLLM: Bool { engine == "mlx" }
    var canStart: Bool {
        guard !name.isEmpty, !isRunning else { return false }
        if isLLM { return !modelRepo.isEmpty && !datasetId.isEmpty }
        return !datasetId.isEmpty  // lm: from-scratch needs data, not a base model
    }
```

In `start()` (the `StartRunRequest(...)` on lines 69-73), pass the new fields:

```swift
        let request = StartRunRequest(
            name: name, modelRepo: modelRepo, datasetId: datasetId, engine: engine,
            fineTuneType: fineTuneType, numLayers: numLayers, batchSize: batchSize, iters: iters,
            learningRate: learningRate, maxSeqLength: maxSeqLength,
            modelSize: modelSize, hiddenSize: hiddenSize, numHeads: numHeads,
            vocabSize: vocabSize, contextLength: contextLength
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd App && xcodebuild test -scheme TinyForge -destination 'platform=macOS' -skipMacroValidation 2>&1 | tail -20`
Expected: PASS (`** TEST SUCCEEDED **`).

- [ ] **Step 6: Commit**

```bash
git add App/Sources/Backend/TrainModels.swift App/Sources/Features/Training/TrainingModel.swift App/Tests/TrainingModelTests.swift
git commit -m "feat(app): lm engine in TrainingModel + StartRunRequest knobs"
```

---

### Task 10: Swift — Training UI for the lm engine

**Files:**
- Modify: `App/Sources/Features/Training/TrainingView.swift`

**Interfaces:** consumes the `TrainingModel` fields from Task 9. No new test (SwiftUI view; covered by a successful build).

- [ ] **Step 1: Replace the engine picker + hint block**

In `App/Sources/Features/Training/TrainingView.swift`, replace the `Picker("Engine", ...)` (lines 30-35) with:

```swift
                Picker("Engine", selection: $model.engine) {
                    Text("LLM LoRA (MLX)").tag("mlx")
                    Text("Tiny LM from scratch").tag("lm")
                }
                .pickerStyle(.segmented)
```

Replace the `} else if model.engine == "vision" { ... } else { ... }` hint branch (lines 62-66) with an `lm` config block:

```swift
                } else {
                    hint("Trains a small Llama-style LM from scratch on your dataset (Apple Silicon GPU). Pick a size; advanced knobs override it.")
                    if model.datasets.isEmpty {
                        hint("Prepare a dataset in the Datasets tab first.")
                    } else {
                        Picker("Dataset", selection: $model.datasetId) {
                            Text("Select…").tag("")
                            ForEach(model.datasets) { Text($0.name).tag($0.id) }
                        }
                    }
                    Picker("Model size", selection: $model.modelSize) {
                        Text("Tiny (~1–3M)").tag("tiny")
                        Text("Small (~8–15M)").tag("small")
                        Text("Medium (~30–60M)").tag("medium")
                        Text("Advanced").tag("custom")
                    }
                    .pickerStyle(.segmented)
                    if model.modelSize == "custom" {
                        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
                            GridRow {
                                stepper("Hidden size", $model.hiddenSize, 64...1024, step: 64)
                                stepper("Layers", $model.numLayers, 1...24)
                            }
                            GridRow {
                                stepper("Heads", $model.numHeads, 1...16)
                                stepper("Context", $model.contextLength, 64...2048, step: 64)
                            }
                            GridRow {
                                stepper("Vocab size", $model.vocabSize, 1000...50000, step: 1000)
                            }
                        }
                    }
                }
```

- [ ] **Step 2: Build to verify it compiles**

Run: `cd App && xcodebuild build -scheme TinyForge -destination 'platform=macOS' -skipMacroValidation 2>&1 | tail -15`
Expected: `** BUILD SUCCEEDED **`.

- [ ] **Step 3: Commit**

```bash
git add App/Sources/Features/Training/TrainingView.swift
git commit -m "feat(app): training UI for the tiny-LM engine (size presets + advanced knobs)"
```

---

### Task 11: Swift — surface run engine + Playground from-scratch models

**Files:**
- Modify: `App/Sources/Backend/TrainModels.swift`
- Modify: `App/Sources/Features/Playground/PlaygroundModel.swift`
- Modify: `App/Sources/Features/Playground/PlaygroundView.swift`
- Modify: `App/Tests/PlaygroundModelTests.swift`

**Interfaces:**
- Consumes: backend `RunRecord.engine` (Task 6).
- Produces: Swift `RunRecord` gains `engine`. `PlaygroundModel` gains `scratchRuns: [RunRecord]` and `selectScratchModel(_ run: RunRecord)` setting `modelRepo = run.adapterPath`, `adapterRunId = ""`.

- [ ] **Step 1: Write failing Playground tests**

In `App/Tests/PlaygroundModelTests.swift`, add (using the file's existing fakes/helpers; mirror how other tests build `RunRecord`):

```swift
    @Test func scratchRunsListsOnlyLmEngineRuns() async {
        let api = FakeBackendAPI()
        api.runs = [
            runRecord(id: "a", state: "completed", engine: "lm", adapterPath: "/runs/a"),
            runRecord(id: "b", state: "completed", engine: "mlx", adapterPath: "/runs/b"),
        ]
        let sut = PlaygroundModel(api: api, infer: FakeInfer())
        await sut.loadInputs()
        #expect(sut.scratchRuns.map(\.id) == ["a"])
    }

    @Test func selectScratchModelPointsModelRepoAtRunDir() {
        let sut = PlaygroundModel(api: FakeBackendAPI(), infer: FakeInfer())
        let run = runRecord(id: "a", state: "completed", engine: "lm", adapterPath: "/runs/a")
        sut.selectScratchModel(run)
        #expect(sut.modelRepo == "/runs/a")
        #expect(sut.adapterRunId == "")
    }
```

If `runRecord(...)`/`FakeInfer` helpers don't already exist in this test file, add a local `runRecord` helper that builds a `RunRecord` with the new `engine` field, matching the decoder/key names.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd App && xcodebuild test -scheme TinyForge -destination 'platform=macOS' -skipMacroValidation 2>&1 | tail -30`
Expected: FAIL — `RunRecord.engine`, `scratchRuns`, `selectScratchModel` don't exist.

- [ ] **Step 3: Add `engine` to Swift `RunRecord`**

In `App/Sources/Backend/TrainModels.swift`, add to `RunRecord` (after `let state: String`, line 39):

```swift
    let engine: String
```

Add to its `CodingKeys` (line 43, the `case id, name, state` line) — include `engine`:

```swift
        case id, name, state, engine
```

Backend always returns `engine` now, but guard against older cached payloads by giving `RunRecord` a defaulting decoder. Add this initializer inside `RunRecord`:

```swift
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        name = try c.decode(String.self, forKey: .name)
        modelRepo = try c.decode(String.self, forKey: .modelRepo)
        datasetId = try c.decode(String.self, forKey: .datasetId)
        state = try c.decode(String.self, forKey: .state)
        engine = try c.decodeIfPresent(String.self, forKey: .engine) ?? "mlx"
        createdAt = try c.decode(String.self, forKey: .createdAt)
        adapterPath = try c.decode(String.self, forKey: .adapterPath)
    }
```

> Note: adding a custom `init(from:)` to a struct that also needs a memberwise initializer for tests means you must add an explicit memberwise `init` too. Add:

```swift
    init(id: String, name: String, modelRepo: String, datasetId: String, state: String,
         engine: String, createdAt: String, adapterPath: String) {
        self.id = id; self.name = name; self.modelRepo = modelRepo; self.datasetId = datasetId
        self.state = state; self.engine = engine; self.createdAt = createdAt; self.adapterPath = adapterPath
    }
```

- [ ] **Step 4: Add scratch-model support to `PlaygroundModel`**

In `App/Sources/Features/Playground/PlaygroundModel.swift`, add after `runs` (line 18 area) a computed list and selection method. Add this computed property near `selectedRun` (line 42):

```swift
    var scratchRuns: [RunRecord] { runs.filter { $0.engine == "lm" } }

    func selectScratchModel(_ run: RunRecord) {
        modelRepo = run.adapterPath  // the run dir IS a loadable HF model
        adapterRunId = ""            // from-scratch model: no adapter overlay
    }
```

(The existing `loadInputs()` already filters `runs` to completed; `scratchRuns` derives from it.)

- [ ] **Step 5: Add a Playground picker for from-scratch models**

In `App/Sources/Features/Playground/PlaygroundView.swift`, add a picker that lists `model.scratchRuns` and calls `model.selectScratchModel`. Place it near the existing base-model selection (match the file's existing `Picker` style). Minimal addition:

```swift
                if !model.scratchRuns.isEmpty {
                    Picker("From-scratch model", selection: Binding(
                        get: { model.modelRepo },
                        set: { id in
                            if let run = model.scratchRuns.first(where: { $0.adapterPath == id }) {
                                model.selectScratchModel(run)
                            } else {
                                model.modelRepo = id
                            }
                        })
                    ) {
                        Text("Downloaded base model").tag(model.modelRepo)
                        ForEach(model.scratchRuns) { Text($0.name).tag($0.adapterPath) }
                    }
                }
```

(If `PlaygroundView` doesn't already display a base-model picker structure that this fits into, add the block inside the same config `VStack`; the exact insertion point follows the file's existing layout.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd App && xcodebuild test -scheme TinyForge -destination 'platform=macOS' -skipMacroValidation 2>&1 | tail -20`
Expected: PASS. Fix any other call sites that construct `RunRecord` (now requiring `engine:`) surfaced by the compiler — update them to pass `engine: "mlx"`.

- [ ] **Step 7: Commit**

```bash
git add App/Sources/Backend/TrainModels.swift App/Sources/Features/Playground/PlaygroundModel.swift App/Sources/Features/Playground/PlaygroundView.swift App/Tests/PlaygroundModelTests.swift
git commit -m "feat(app): select from-scratch lm runs as Playground models"
```

---

### Task 12: Swift — Export UI drops gguf/coreml

**Files:**
- Modify: `App/Sources/Features/Export/ExportView.swift`
- Modify: `App/Tests/ExportModelTests.swift` (only if it references gguf/coreml)

**Interfaces:** none new; the export request `target` is a free `String` in Swift, so this is UI-only.

- [ ] **Step 1: Remove gguf/coreml from the format picker**

In `App/Sources/Features/Export/ExportView.swift`, replace the `Picker("Format", ...)` block (lines 21-31) with:

```swift
                        Picker("Format", selection: $model.target) {
                            Text("safetensors").tag("safetensors")
                            Text("MLX").tag("mlx")
                        }
                        .pickerStyle(.segmented)
```

(This deletes the `GGUF`/`Core ML` tags and the `coreml` helper-text `if` block.)

- [ ] **Step 2: Check the export tests for stale targets**

Run: `cd App && grep -n "gguf\|coreml" App/Tests/ExportModelTests.swift`
Expected: if any hits, change those tests to use `"safetensors"` or `"mlx"`. If none, no change.

- [ ] **Step 3: Run tests to verify they pass**

Run: `cd App && xcodebuild test -scheme TinyForge -destination 'platform=macOS' -skipMacroValidation 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add App/Sources/Features/Export/ExportView.swift App/Tests/ExportModelTests.swift
git commit -m "feat(app): export UI offers safetensors + MLX only"
```

---

### Task 13: Docs — README + spec alignment

**Files:**
- Modify: `README.md`

**Interfaces:** none.

- [ ] **Step 1: Update the feature copy**

In `README.md`, in the Features section (around lines 37-39):
- Replace the finetuning bullet's "a from-scratch **PyTorch/MPS** engine, and a **vision** engine (ViT…)" clause with: "and a **from-scratch tiny-LM** engine (a small Llama-style model trained on your own text)".
- In the Export bullet, change the targets list to "**safetensors**, **MLX (quantized)**" and remove "**GGUF**, or **Core ML** (`.mlpackage`)" (or move them under a "planned" note).

In the `train/` line of the Project structure (line 113), keep "mlx-lm & torch runners" accurate: change to "mlx-lm & from-scratch LM runners".

- [ ] **Step 2: Verify no stale demo references remain**

Run: `grep -rn "synthetic\|from-scratch MLP\|ViT image classifier\|torch_worker\|vision_worker" README.md docs/`
Expected: only the design/plan spec files under `docs/superpowers/` may mention the old engines (historical); README is clean.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: describe the from-scratch tiny-LM engine; trim export targets"
```

---

### Task 14: Full-suite verification

**Files:** none (verification only).

- [ ] **Step 1: Backend suite**

Run: `cd backend && uv run pytest -q`
Expected: all PASS (heavy `lm_worker` smoke test skipped unless `.run-network-tests` exists).

- [ ] **Step 2: App suite**

Run: `cd App && xcodebuild test -scheme TinyForge -destination 'platform=macOS' -skipMacroValidation 2>&1 | tail -20`
Expected: `** TEST SUCCEEDED **`.

- [ ] **Step 3: Optional real end-to-end**

```bash
touch .run-network-tests
cd backend && uv run pytest tests/test_lm_worker.py -q
rm ../.run-network-tests
```
Expected: the smoke test trains a few steps, asserts loss is printed, and that `model.safetensors`/`config.json`/`tokenizer.json` are written.

- [ ] **Step 4: Final commit (if any verification fixups were needed)**

```bash
git add -A && git commit -m "test: green backend + app suites for the lm engine" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage:**
- New `lm` engine (HF Trainer + Llama) → Tasks 1, 4. ✓
- Corpus BPE tokenizer → Task 3. ✓
- Data rendering + packing → Task 2. ✓
- Live dashboards via existing parser → Task 4 prints parser-matching lines; reuse unchanged. ✓
- HF-format checkpoint output → Task 4 (`save_pretrained` + tokenizer). ✓
- Playground generation → Task 11. ✓
- Export safetensors + MLX, full-model branch → Task 7. ✓
- Remove torch/vision/coreml → Task 8 (+ Tasks 5/7 removed their wiring). ✓
- Presets + advanced → Tasks 1, 9, 10. ✓
- Service `data_dir` always resolved + `model_repo = adapter_path` for lm → Task 6. ✓
- Error handling (empty/small corpus) → Task 4 (`SystemExit` on empty train / zero blocks). ✓
- README/docs update → Task 13. ✓
- Tests (unit + opt-in integration) → each task; full suite Task 14. ✓

**Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N". Every code step shows complete code. The two view-insertion notes (Playground/Export) give exact code and an explicit "follow existing layout" instruction, not a placeholder.

**Type consistency:** `apply_preset` returns `(num_layers, hidden_size, num_heads, context_length)` and is consumed in that order in Task 6. `RunResolver` 3-tuple `(model_repo, adapter_path, engine)` defined in Task 7 and produced by `services.py` in the same task. Swift `RunRecord.engine` added in Task 11 and consumed by `scratchRuns`/`selectScratchModel` in the same task; Task 9's `StartRunRequest` field names (`modelSize`, `hiddenSize`, `numHeads`, `vocabSize`, `contextLength`) match their coding keys and `TrainingModel` usage.

**Known follow-up (not blocking):** GGUF and Core ML export remain deferred per the spec.
