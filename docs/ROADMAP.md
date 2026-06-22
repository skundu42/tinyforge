# TinyForge roadmap

Milestones are each independently shippable and verifiable. M0–M5 build the LLM
finetuning product; M6 adds breadth; M7 is packaging/polish.

| Milestone | Scope | Status |
|-----------|-------|--------|
| **M0** | Foundations / walking skeleton: SwiftUI app ⇄ embedded Python FastAPI sidecar, process lifecycle, health/runtime | ✅ done |
| **M1** | HuggingFace Hub browse/download + cache + token auth | ✅ done |
| **M2** | Dataset builder (import, format, tokenize preview, splits) | ✅ done |
| **M3** | LLM LoRA finetuning (MLX) + live dashboards + experiment tracking | ✅ done |
| **M4** | Inference playground (base vs finetuned, streaming) | ✅ done |
| **M5** | Exports (safetensors, MLX, GGUF) + push to Hub | ✅ done |
| **M6** | PyTorch/MPS engine (from-scratch) integrated into the run system | ✅ done |
| **M7** | Packaging: bundled Python + signed/notarized DMG | ✅ done |

## M0 — delivered

**Architecture.** Native SwiftUI app spawns a bundled Python FastAPI sidecar bound
to `127.0.0.1` on an OS-chosen ephemeral port, authenticated with a per-launch
bearer token. The backend prints a single JSON `ready` line on stdout (port);
all logs go to stderr.

**Backend (`backend/`, Python 3.13, uv):**
- `tinyforge.api` — `create_app(token)` with public `/health` and token-guarded `/v1/runtime`.
- `tinyforge.server` — ephemeral loopback bind, ready-line protocol, uvicorn on the bound socket.
- `tinyforge.system` — interpreter + engine-availability introspection.
- `tinyforge.watchdog` — parent-death watchdog (self-terminates if the host app dies; no PR_SET_PDEATHSIG on macOS).
- 16 pytest tests.

**App (`App/`, SwiftUI, Swift 6, XcodeGen):**
- `BackendProcessManager` — actor that spawns the backend, drains stderr, races the ready line against a timeout, tracks the child PID.
- `APIClient` — typed REST client over an injectable `Transport`.
- `ReadyLineParser`, `BackendLauncher` (env / DEBUG-self-locate / bundled runtime resolution), `BackendController` (`@Observable`), `ContentView`.
- `AppDelegate.applicationWillTerminate` + the Python watchdog give two-sided orphan prevention.
- 11 Swift Testing tests incl. an end-to-end test that spawns the real backend.

**Verified:** clean app quit and force-kill (SIGKILL) both reap the backend (no orphans); full handshake (spawn → ready → `/health` → `/v1/runtime`) green.

## M1 — delivered

Full HuggingFace Hub integration, native end-to-end.

- **Backend** (`tinyforge/hub/`, `services.py`, `api/routers/hub.py`): token auth
  (HF_TOKEN-priority, login/whoami/logout), model/dataset search + detail, a
  threaded download manager with real byte progress (per-file `hf_hub_download`
  byte bars — `snapshot_download`/Xet don't surface incremental bytes), cache
  scan/delete, REST routes + a token-guarded WebSocket progress stream.
- **App** (`Features/Hub`, `Features/Settings`): `APIClient` hub methods over the
  typed `BackendAPI`, a WebSocket `DownloadProgressClient`, a `NavigationSplitView`
  shell, a Hub browser (search · sort · results · detail · download with live
  progress), and a Settings panel (HF token sign-in · cache management).
- **Verified e2e** (opt-in network test): real Swift client → live backend →
  Hub search + download with WebSocket progress to completion. Caught + fixed a
  WebSocket token bug (base64 `+` mangled in the query string → switched to a
  URL-safe hex token).

Backend: 45 pytest tests. App: 25 Swift Testing tests.

## M2 — delivered

A dataset builder that turns raw data into mlx-lm-ready splits.

- **Backend** (`tinyforge/datasets/`): load HF or local JSON/CSV/Parquet
  (datasets v5) + preview; formatting into mlx-lm `text` / `completion` /
  `chat` (built-in templates incl. Alpaca); tokenization preview + length
  histogram (via `tokenizers`, no torch); deterministic train/val split; a
  JSONL + SQLite registry of prepared datasets; `/v1/datasets` routes.
  App data under `~/Library/Application Support/TinyForge/` (overridable via
  `TINYFORGE_DATA_DIR`).
- **App** (`Features/Datasets`): a builder UI — source (Hub/local), preview
  table, format + column mapping, optional token-length analysis (Swift
  Charts histogram), validation-split slider, and a registry of prepared
  datasets.
- **Verified e2e** (offline, hermetic): real client → backend → local JSONL →
  preview + Alpaca prepare → registered `train.jsonl` in `completion` format.

Backend: 71 pytest tests. App: 30 Swift Testing tests.

## M3 — delivered

MLX LoRA/QLoRA finetuning with live dashboards and experiment tracking.

- **Backend** (`tinyforge/train/`): `mlx_lm.lora` run as an isolated subprocess;
  output parsed into structured train/val/saved events mirrored to
  `events.jsonl`; `RunConfig`/`build_command`; `TrainingRunner` (start/stop/
  status/events); SQLite `RunRegistry`; `TrainingService` resolves a dataset id
  to its prepared dir, assigns an output dir, persists records, syncs live
  state; `/v1/runs` routes + a token-guarded WebSocket event stream.
  Installed mlx 0.31.2 / mlx-lm 0.31.3 / transformers 5.12.1.
- **App** (`Features/Training`, `Telemetry`): pick a cached model + prepared
  dataset + hyperparameters, start a run, watch live Swift Charts dashboards
  (train/val loss, throughput, peak memory) with GPU-budget + thermal
  telemetry, stop, and browse run history. `RunEventClient` streams events.
- **Verified e2e**: real client → backend → `mlx_lm.lora` → WebSocket; a 3-iter
  LoRA finetune on the cached SmolLM streams metrics to `completed` and saves
  an adapter (≈2s).

Backend: 96 pytest tests. App: 35 Swift Testing tests.

## M4 — delivered

An inference playground that streams generations, base vs finetuned.

- **Backend** (`tinyforge/infer/`): `InferenceService` streams tokens via mlx_lm
  (`load` with optional `adapter_path`, `stream_generate`, sampler from
  temp/top-p, chat templating); single-model cache evicted on switch;
  `/v1/infer` WebSocket runs the sync generator in a thread and streams
  token/done/error.
- **App** (`Features/Playground`): pick a cached model + an optional finetuned
  adapter (a completed run), set sampling controls, and stream output —
  side-by-side **Base vs Finetuned** when an adapter is selected.
  `InferenceClient` drives the WebSocket.
- **Verified e2e**: real client → backend → mlx_lm streaming generation over the
  WebSocket from the cached model.

Backend: 100 pytest tests. App: 39 Swift Testing tests.

## M5 — delivered

Export a finetune to a standalone model and share it.

- **Backend** (`tinyforge/export/`): `ExportManager` fuses the LoRA adapter into
  its base (`mlx_lm.fuse`) → **safetensors** (HF format); **MLX-quantized**
  (`mlx_lm.convert -q`, configurable bits); **GGUF** (`fuse --gguf-path`, for
  supported architectures); optional **push to Hub** (`upload_folder` + auto
  `ModelCard`). Runs as a job; injectable command runner / pusher / run
  resolver. `/v1/exports` routes.
- **App** (`Features/Export`): pick a completed run, choose a format (+ quant
  bits for MLX), optionally enter a Hub repo to push to, run the export, and
  see export history with output paths + Hub links.
- **Verified**: a real fuse → safetensors export produces a complete HF-format
  model (`model.safetensors` + `config.json` + tokenizer).

> Core ML export is deferred: converting MLX/HF weights via coremltools needs a
> traceable PyTorch path and is a larger effort; planned for a later pass.

Backend: 111 pytest tests. App: 44 Swift Testing tests.

## M6 — delivered

A second engine — PyTorch on the MPS GPU — proving the dual-engine design.

- **Backend**: a `torch` engine option on runs. `torch_worker` trains a small
  MLP on a synthetic non-linear task on the **MPS device** (torch 2.12,
  `mps.is_available()`), printing progress in the **same event format** the mlx
  parser understands — so it reuses the entire run system (registry, WebSocket
  stream, live Swift Charts dashboards). `build_command` branches by engine;
  `TrainingService` skips model/dataset resolution for from-scratch runs.
- **App**: a Finetune engine selector — *LLM LoRA (MLX)* vs *From-scratch
  (PyTorch/MPS)* — that hides the model/dataset pickers for from-scratch runs.
- **Verified**: a real torch/MPS run completes with loss decreasing
  (0.75 → 0.05) and saves `model.pt`, streamed through the existing dashboards.

> This establishes the engine + event contract. HF Trainer (vision/audio) and
> TRL (SFT/DPO) workers plug in as additional engines emitting the same events;
> from-scratch on MPS is the representative path built and verified here.

Backend: 112 pytest tests. App: 45 Swift Testing tests.

## M7 — delivered

A self-contained, distributable app.

- **`scripts/bundle_python.sh`**: a relocatable python-build-standalone CPython
  (~1 GB with torch/mlx/transformers) installed directly into its site-packages;
  the app launches it via `Contents/Resources/python/bin/python3 -m tinyforge`.
- **`scripts/sign.sh`**: inside-out Developer-ID signing (not `--deep`) with
  hardened-runtime entitlements (`disable-library-validation`,
  `allow-unsigned-executable-memory`, `allow-jit`).
- **`scripts/package_dmg.sh`** + **`scripts/notarize.sh`**: signed DMG +
  notarytool submit/staple. **`scripts/build_release.sh`** orchestrates the lot.
- **Verified**: the bundled runtime starts the backend standalone, and a Release
  app spawns the **bundled** interpreter — no dev-toolchain dependency.
  (Notarization itself needs your Apple ID credentials; see `docs/packaging.md`.)

App: 45 Swift Testing tests. Backend: 112 pytest tests.

## Conventions
- TDD: tests first (see `backend/tests/`, `App/Tests/`).
- Backend deps pinned via `uv.lock`; the Xcode project is generated from `App/project.yml` (`xcodegen generate`).
- Loopback-only API + per-launch token; training/export jobs will run as isolated worker subprocesses (M3+).
