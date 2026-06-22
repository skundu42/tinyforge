<div align="center">

<img src="docs/screenshots/logo.png" width="128" alt="TinyForge logo" />

# TinyForge

**Train, finetune, and experiment with tiny ML models — entirely on your Mac.**

A native macOS studio for the whole local ML loop: browse a model, build a dataset, finetune it on your Apple Silicon GPU, try it out, and export it — no cloud, no notebooks, no toolchain to set up.

[![Platform](https://img.shields.io/badge/platform-macOS%20·%20Apple%20Silicon-1f1147?style=flat-square)](#requirements)
[![Swift](https://img.shields.io/badge/Swift-6.3-F05138?style=flat-square&logo=swift&logoColor=white)](https://swift.org)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-6E56CF?style=flat-square)](https://github.com/ml-explore/mlx)
[![Tests](https://img.shields.io/badge/tests-158%20passing-2FB67C?style=flat-square)](#testing)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)

<img src="docs/screenshots/home.png" width="760" alt="TinyForge home — the forge journey" />

</div>

---

## Why TinyForge

Finetuning a small model on your own data shouldn't require a GPU rental, a stack of Python scripts, and three different CLIs. TinyForge wraps the best of the Apple Silicon ML ecosystem — **MLX**, **mlx-lm**, **PyTorch/MPS**, and the **HuggingFace** stack — behind a clean, native app that walks you through one coherent workflow:

> **Get a model → Build a dataset → Finetune → Try it out → Export & share**

Everything runs locally. Your data never leaves your machine.

## Features

- 🔍 **Model browser** — search HuggingFace, view files & READMEs, and download with a live progress bar (Xet-accelerated). See what you've already downloaded at a glance.
- 🧱 **Dataset builder** — import HuggingFace datasets or local JSON/CSV/Parquet, preview rows, map columns into chat / instruction / completion formats, inspect token-length distributions, and split into train/validation.
- ⚡ **Finetuning on the GPU** — LoRA / QLoRA / DoRA / full finetuning via **mlx-lm**, plus a from-scratch **PyTorch/MPS** engine — with **live dashboards** (loss, throughput, peak memory) and GPU/thermal telemetry.
- ✨ **Inference playground** — stream generations with sampling controls and compare **base vs. finetuned** side by side.
- 📦 **Export & share** — fuse adapters and export to **safetensors**, **MLX (quantized)**, or **GGUF**, and push straight to the Hub with an auto-generated model card.
- 🗂️ **Experiment tracking** — every run, dataset, and export is recorded locally (SQLite) and browsable.
- 📥 **Self-contained** — ships as a signed, notarizable `.app` with a bundled Python runtime. Nothing to install.

## Screenshots

<div align="center">
<table>
  <tr>
    <td><img src="docs/screenshots/models.png" alt="Models" /><br/><sub><b>Models</b> — browse & manage downloads</sub></td>
    <td><img src="docs/screenshots/datasets.png" alt="Datasets" /><br/><sub><b>Datasets</b> — build training data</sub></td>
  </tr>
  <tr>
    <td><img src="docs/screenshots/finetune.png" alt="Finetune" /><br/><sub><b>Finetune</b> — configure & watch it train</sub></td>
    <td><img src="docs/screenshots/playground.png" alt="Playground" /><br/><sub><b>Playground</b> — base vs. finetuned</sub></td>
  </tr>
</table>
</div>

## Requirements

- **Apple Silicon** Mac (M1 or later) — MLX and the MPS backend require it.
- **macOS 15+**.
- For development: **Xcode 26+**, [`xcodegen`](https://github.com/yonsm/XcodeGen), and [`uv`](https://github.com/astral-sh/uv).

## Quick start

### Run a release build

```bash
git clone <repo-url> tinyforge && cd tinyforge
scripts/build_release.sh          # bundles Python, builds, signs → TinyForge.app
open build/Build/Products/Release/TinyForge.app
```

### Run from source (development)

```bash
# 1) Backend deps
cd backend && uv sync && cd ..

# 2) Generate the Xcode project and run
cd App && xcodegen generate && open TinyForge.xcodeproj   # then ⌘R
```

In a debug build, the app finds the dev Python environment automatically — no bundling needed.

## Architecture

TinyForge is a native **SwiftUI** app talking to an embedded **Python** ML service over a loopback API. Heavy or crash-prone work (training, exports) runs as **isolated worker subprocesses**, so a native segfault never takes down the UI.

```
┌──────────────────────────────────────────────┐
│  SwiftUI app (Swift 6, strict concurrency)    │
│  Home · Models · Datasets · Finetune ·        │
│  Playground · Export · Settings               │
│  Metal / thermal telemetry                    │
└───────────────▲───────────────────┬───────────┘
   REST + WebSocket                 │ spawn / lifecycle
   (127.0.0.1, ephemeral port,      │ (swift Process, watchdog)
    per-launch bearer token)        ▼
┌──────────────────────────────────────────────┐
│  Python orchestrator (FastAPI / uvicorn)      │
│  hub · datasets · train · infer · export      │
│  SQLite registries                            │
└───────────────┬──────────────────────────────┘
   spawns one isolated child per job
   ┌────────────┴───────────────┐
   ▼                            ▼
 mlx_lm.lora                  torch (MPS)
 fuse / convert / generate    from-scratch
```

The whole thing ships self-contained: a relocatable [python-build-standalone](https://github.com/astral-sh/python-build-standalone) interpreter (with torch/mlx/transformers) is bundled inside the `.app` and signed inside-out for notarization. See [`docs/packaging.md`](docs/packaging.md).

## Tech stack

| Layer | Tools |
|-------|-------|
| **App** | SwiftUI · Swift 6.3 · Swift Charts · swift-subprocess · XcodeGen |
| **Backend** | FastAPI · uvicorn · pydantic · uv |
| **ML** | MLX · mlx-lm · PyTorch (MPS) · transformers · datasets · tokenizers |
| **Hub** | huggingface_hub (Xet) |
| **Packaging** | python-build-standalone · codesign · notarytool |

## Project structure

```
.
├── App/                  # SwiftUI macOS app (project.yml → Xcode project)
│   ├── Sources/
│   │   ├── Backend/      # process manager, API client, WebSocket clients
│   │   ├── DesignSystem/ # theme + reusable components
│   │   ├── Features/     # Home, Hub, Datasets, Training, Playground, Export, Settings
│   │   └── Telemetry/
│   └── Tests/
├── backend/              # Python orchestrator (uv project)
│   └── tinyforge/
│       ├── api/          # FastAPI app + routers
│       ├── hub/          # HuggingFace browse / download / cache / auth
│       ├── datasets/     # load / format / tokenize / registry
│       ├── train/        # mlx-lm & torch runners, orchestration, registry
│       ├── infer/        # streaming generation
│       └── export/       # fuse / convert / push
├── scripts/              # bundle_python, sign, notarize, package_dmg, build_release
└── docs/                 # roadmap, packaging guide, screenshots
```

## Testing

The project is built test-first. Logic is unit-tested with fakes; each milestone is verified end-to-end against the real toolchain (live HuggingFace, a real MLX finetune, a real MPS run, a real fuse, a bundled-runtime launch).

```bash
# Backend (112 tests)
cd backend && uv run pytest

# App (46 tests)
cd App && xcodebuild test -scheme TinyForge -destination 'platform=macOS'
```

Network/heavy end-to-end tests are opt-in: `touch .run-network-tests` to enable them.

## Roadmap

| Milestone | Status |
|-----------|--------|
| M0 — Foundations (SwiftUI ⇄ embedded Python) | ✅ |
| M1 — HuggingFace Hub browse/download | ✅ |
| M2 — Dataset builder | ✅ |
| M3 — MLX LoRA finetuning + live dashboards | ✅ |
| M4 — Inference playground | ✅ |
| M5 — Exports (safetensors / MLX / GGUF) + push to Hub | ✅ |
| M6 — PyTorch/MPS engine | ✅ |
| M7 — Bundled Python + notarized DMG | ✅ |
| Core ML export · HF Trainer (vision/audio) · native MLX-Swift inference | 🔜 |

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for details on what each milestone delivered.

## Contributing

Contributions are welcome! A good loop:

1. `cd backend && uv sync` and `cd App && xcodegen generate`.
2. Make your change **test-first** — add a failing test, then the code.
3. Run `uv run pytest` and `xcodebuild test` — keep them green.
4. Keep the design system (`App/Sources/DesignSystem`) consistent for any UI.
5. Open a PR with a clear description and, for UI work, a screenshot.

Found a bug or have an idea? Please open an issue.

## License

Released under the [MIT License](LICENSE). The TinyForge name and logo are part of this project; the bundled dependencies retain their own licenses.

## Acknowledgements

Built on the shoulders of [MLX](https://github.com/ml-explore/mlx) & [mlx-lm](https://github.com/ml-explore/mlx-lm), [PyTorch](https://pytorch.org), [HuggingFace](https://huggingface.co), [FastAPI](https://fastapi.tiangolo.com), and [uv](https://github.com/astral-sh/uv).

<div align="center"><sub>Forged on Apple Silicon. 🔨✨</sub></div>
