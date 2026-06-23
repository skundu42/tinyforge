# Tiny LM From-Scratch Training Engine — Design

**Date:** 2026-06-23
**Status:** Approved for planning
**Topic:** Replace the synthetic toy training engines with a real from-scratch tiny-LM trainer that round-trips through Playground and Export.

---

## 1. Problem

TinyForge advertises "Train, finetune, and experiment with tiny ML models." Today only the **`mlx`** engine does real work (LoRA/full finetuning of a pretrained model via `mlx-lm`). The other two engines are toys:

- `train/torch_worker.py` — a fixed 3-layer MLP trained on **synthetic random vectors**; ignores the user's dataset entirely.
- `train/vision_worker.py` — a freshly-initialized tiny ViT trained on **random 32×32 noise images**; ignores the user's dataset entirely.

Neither trains on user data. This spec adds a genuine **from-scratch language-model trainer** (`lm` engine) and **removes both toy engines**.

## 2. Goal & scope

**Goal:** A new `lm` training engine that trains a small Llama-style causal LM **from random initialization** on the user's **real text dataset**, with live dashboards, and that round-trips through the existing Playground (generation) and Export (safetensors + MLX-quantized).

**In scope (v1):**
1. Train a BPE tokenizer on the user's corpus (truly from scratch; small vocab).
2. Randomly initialize a `LlamaForCausalLM` and train it with HuggingFace `Trainer` on tokenized, packed text.
3. Live train/val loss, throughput, peak-mem dashboards — reuse the existing event contract unchanged.
4. Save a standard HuggingFace checkpoint dir (`config.json` + `model.safetensors` + tokenizer files).
5. **Playground** generation from the trained model.
6. **Export** to safetensors (the saved dir) and MLX-quantized (`mlx_lm convert`).
7. **Remove** the `torch` and `vision` toy engines and all their wiring, tests, and UI.

**Out of scope (v1), deferred:**
- GGUF and Core ML export of the LM (Core ML for an autoregressive LM with KV-cache is a project on its own).
- Multi-epoch curricula, run resumption, gradient accumulation tuning beyond a sane default, distributed training.

## 3. Approach (decided)

- **New `lm` engine** alongside `mlx`. The `torch` and `vision` engines are deleted.
- **HF Trainer + `LlamaForCausalLM`**, mirroring the (now-removed) `vision_worker.py` pattern: a `TrainerCallback.on_log` prints event lines in the exact format the existing `train/parser.py` understands, so the runner, parser, run registry, WebSocket stream, and Swift dashboards are reused **unchanged**.
- **Corpus-trained BPE tokenizer** (HF `tokenizers`), default vocab ~8k (configurable), saved as `tokenizer.json` + `tokenizer_config.json` so the model round-trips in `mlx_lm` and `transformers`.
- **Standard HF-format output** so Playground (`mlx_lm.load(<dir>)`) and Export (`mlx_lm convert --hf-path <dir>`) work with minimal new code.

Rationale: all three candidate approaches (HF-library, hand-rolled nanoGPT, MLX-native) are genuinely from-scratch. The HF-library route adds the least new integration code given the "full round-trip" requirement, and reuses a pattern already proven in the codebase.

## 4. Components

### 4.1 New backend modules

**`backend/tinyforge/train/lm_data.py`** — dataset → packed token tensors.
- Read `train.jsonl` / `valid.jsonl` from `data_dir` (the prepared dataset the existing dataset builder writes).
- Render each row to plain text: `text` rows use their text; `completion`/`alpaca` rows concatenate prompt+completion; `chat`/`messages` rows flatten roles into a single string. (Reuses the formats already produced by `datasets/formatting.py`.)
- Train or load the tokenizer (delegates to `tokenizer.py`).
- Tokenize, concatenate documents with an EOS separator, and chunk into fixed `context_length` blocks (packed pretraining). Produce a `torch.utils.data.Dataset` of `input_ids`/`labels` (labels = input_ids for causal LM).
- Pure/testable: no Trainer, no subprocess.

**`backend/tinyforge/train/tokenizer.py`** — corpus BPE tokenizer.
- Train a byte-level BPE (`tokenizers`) on an iterator of corpus strings, target `vocab_size`, with special tokens (`<|endoftext|>`/EOS, pad).
- Wrap as `PreTrainedTokenizerFast` and `save_pretrained` so it loads in `transformers`/`mlx_lm`.
- Testable on a tiny in-memory string (train + encode/decode round-trip).

**`backend/tinyforge/train/lm_worker.py`** — the worker process (entry point, like the old vision worker).
- Args: `--adapter-path` (run output dir), `--data`, `--iters`, `--learning-rate`, `--batch-size`, `--steps-per-report`, `--seed`, plus model knobs `--hidden-size`, `--num-layers`, `--num-heads`, `--vocab-size`, `--context-length`.
- Build dataset via `lm_data`, build `LlamaConfig(...)` from knobs (`num_key_value_heads = num_heads` for v1 simplicity), random-init `LlamaForCausalLM(config)`.
- `Trainer.train()` with a `TrainerCallback` that prints `Iter N: Train loss … Learning Rate … It/sec … Tokens/sec … Trained Tokens … Peak mem … GB` and `Iter N: Val loss …` lines (parser-compatible). Tokens are **real** here, so these metrics are finally accurate.
- On finish: `trainer.save_model(adapter_path)` + `tokenizer.save_pretrained(adapter_path)`; print the existing `Saved final weights to …` line so the "saved" event fires.

### 4.2 Changed backend modules

**`train/models.py`**
- `engine` Literal becomes `Literal["mlx", "lm"]` (drop `torch`, `vision`).
- Add LM knobs to `RunConfig` and `StartRunRequest`: `model_size` (preset: `tiny|small|medium|custom`), `hidden_size`, `num_heads`, `vocab_size`, `context_length`. Reuse existing `num_layers`, `iters`, `learning_rate`, `batch_size`. Presets expand to concrete knobs server-side.

**`train/config.py`**
- Replace the `torch`/`vision` branch with an `lm` branch that builds the `lm_worker` command from the LM knobs. Keep the `mlx` branch unchanged.

**`train/service.py`**
- All remaining engines need a dataset, so always resolve `data_dir = self._resolve_dataset(request.dataset_id)` (drop the `"(none)"` special-case and the `default_names` map).
- For `lm`, set `model_repo = adapter_path` (the run output dir) so Playground/Export can locate the from-scratch model. For `mlx`, keep `model_repo = request.model_repo`.
- Expand `model_size` preset → concrete `hidden_size/num_layers/num_heads/context_length` before constructing `RunConfig`.

**Model-size presets** (approximate, final numbers tuned during implementation):

| Preset  | hidden | layers | heads | context | ~params |
|---------|--------|--------|-------|---------|---------|
| tiny    | 128    | 4      | 4     | 256     | ~1–3M   |
| small   | 256    | 6      | 8     | 512     | ~8–15M  |
| medium  | 512    | 8      | 8     | 512     | ~30–60M |
| custom  | user-specified via advanced override                |

### 4.3 Export rework (required by "remove demos" + full-model round-trip)

The current `export/manager.py` **always fuses** (`build_fuse_command`, base-repo + LoRA adapter) before converting, and Core ML only handles the toy `model.pt` / vision dirs. A from-scratch **full** model has no base+adapter, so:

- `export/models.py`: narrow `ExportRequest.target` to `Literal["safetensors", "mlx"]` (drop `gguf`, `coreml` for v1).
- `export/manager.py`: branch on whether the run is a **full model** (from-scratch `lm`) vs a **LoRA adapter** (`mlx`):
  - **Full model** → no fuse. `safetensors`: the run dir is already safetensors — copy/point it into the export dir. `mlx`: `mlx_lm convert --hf-path <run_dir> --mlx-path <out> -q --q-bits N`.
  - **LoRA adapter** (`mlx`) → existing fuse-then-(optional convert) path, unchanged.
  - Drop the `coreml` branch and `coreml_fn`/`_default_coreml`.
- `RunResolver` (or the run record) must expose enough to tell full-vs-adapter (e.g. the run's `engine`, or a resolved `(model_path, adapter_path_or_None)`).
- **Delete** `export/coreml.py`.

### 4.4 Frontend (Swift)

- `Features/Training/TrainingModel.swift`: engine picker `mlx | lm`; `canStart` for `lm` requires `datasetId` but **not** `modelRepo`; add `model_size` preset picker + advanced knobs (hidden/layers/heads/context/vocab); update `isLLM`/hints. Remove `torch`/`vision`.
- `Features/Training/TrainingView.swift`: LM config fields; hint describing real from-scratch LM training. Remove the synthetic-task hints.
- `Backend/TrainModels.swift` (`StartRunRequest`): mirror the new fields; default `engine` stays `mlx`.
- `Features/Export/*`: remove `gguf`/`coreml` from the export-target UI.
- **Playground**: surface completed `lm` runs as selectable models — pass the run's output dir as `model_repo`, `adapter_path = nil`. (Confirm how the Playground currently lists finetuned runs; wire the new case in.)

### 4.5 Removals ("remove all demo stuff")

- Delete `train/torch_worker.py`, `train/vision_worker.py`, `export/coreml.py`.
- Remove their references in `config.py`, `service.py`, `manager.py`, `models.py`.
- Remove/replace their tests: `test_train_config`, `test_train_service`, `test_export_*`, and any parser/runner tests asserting torch/vision specifics.
- Update `README.md`: replace the "from-scratch PyTorch/MPS engine" and "vision engine" bullets with the real `lm` engine; remove GGUF/Core ML from the export list (or mark as planned).

## 5. Data flow

```
Dataset builder (existing) → data_dir/{train,valid}.jsonl (text)
  → lm_data: render rows→text, train BPE tokenizer, tokenize + pack to context_length
  → lm_worker: random-init LlamaForCausalLM(config) → Trainer.train()
       emits "Iter N: Train loss … Tokens/sec … Peak mem" + "Val loss" (parser reuse)
       on finish: save_pretrained(adapter_path) + tokenizer.save_pretrained → "Saved final weights…"
  → run output dir: config.json + model.safetensors + tokenizer.json + tokenizer_config.json
  → Playground: mlx_lm.load(output_dir) → stream_generate           [near-free]
  → Export: safetensors = the dir; mlx = mlx_lm convert --hf-path dir [reworked manager]
```

## 6. Error handling

- **Empty/too-small corpus** (fewer than one full `context_length` block after packing): fail fast in `lm_data` with a clear message before spawning training.
- **vocab_size > distinct corpus tokens**: `tokenizers` handles a smaller realized vocab; log the realized size.
- **MPS unavailable**: fall back to CPU (Trainer handles device); the existing peak-mem line reports `0.0` on CPU (acceptable, pre-existing behavior).
- **Worker crash**: already isolated by `runner.py` (subprocess) → run state `failed` with captured output.
- **Export of a non-existent/incomplete run dir**: manager fails the job with the captured error (existing `_fail` path).

## 7. Testing (TDD)

**Unit (fast, with fakes):**
- `lm_data`: row→text rendering for each format; packing (concat + EOS + fixed-length chunking); val split; empty-corpus error.
- `tokenizer`: train on a tiny string, encode/decode round-trip, special tokens present.
- `config.build_command`: `engine="lm"` produces the expected `lm_worker` arg vector; `mlx` unchanged.
- `service`: `lm` resolves `data_dir`, sets `model_repo = adapter_path`, expands presets.
- `export/manager`: full-model branch skips fuse; `safetensors` points at run dir; `mlx` invokes `convert --hf-path`; LoRA path unchanged. Target Literal rejects `gguf`/`coreml`.

**Integration (opt-in via `.run-network-tests`, heavy):**
- Train ~50 steps on a few hundred lines of real text → train loss strictly decreases.
- Output dir loads in `mlx_lm` and `stream_generate` yields non-empty text.
- `mlx_lm convert` on the output dir produces a loadable quantized dir.

## 8. Rough effort

Comparable to the existing `mlx` engine's footprint — not a one-file change. Largest pieces: `lm_worker` + `lm_data` + `tokenizer` (backend core) and the export-manager rework. Swift UI + Playground wiring is moderate. Detailed sequencing goes in the implementation plan.

## 9. Open questions resolved

1. Toy engines → **removed entirely**.
2. v1 export targets → **safetensors + MLX-quantized** only; GGUF/Core ML deferred.
3. Model sizing → **presets (tiny/small/medium) + advanced override**.
