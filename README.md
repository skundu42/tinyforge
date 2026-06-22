# TinyForge

A native macOS app for training, finetuning, and experimenting with tiny ML models on Apple Silicon — with deep HuggingFace integration.

- **Frontend:** native SwiftUI app (macOS 26+, Apple Silicon).
- **Backend:** embedded Python ML service (FastAPI sidecar) bundled inside the `.app`.
- **Engines:** MLX / mlx-lm (Apple-optimized LLM/VLM LoRA/QLoRA) + PyTorch/MPS via HuggingFace Trainer/TRL (vision, audio, from-scratch, preference methods).
- **HuggingFace:** browse & download (Xet), token auth + gated/private repos, push models + LoRA adapters + auto model cards.
- **Exports:** safetensors + LoRA · GGUF (llama.cpp/Ollama) · Core ML (`.mlpackage`) · MLX weights.
- **Workbench:** dataset builder · live training dashboards · inference playground · experiment tracking.
- **Distribution:** code-signed + notarized DMG.

## Architecture

```
SwiftUI app  ──REST + WebSocket (127.0.0.1, ephemeral port, per-launch token)──▶  Python orchestrator (FastAPI)
   │                                                                                      │
   └── manages backend process lifecycle (swift-subprocess)                               └── spawns isolated worker subprocess per job
```

System telemetry (GPU memory, thermal, power) is gathered natively in Swift (Metal / IOReport / `ProcessInfo.thermalState`). Training/export jobs run as isolated child processes so a native crash never takes down the UI.

## Repository layout

| Path | Purpose |
|------|---------|
| `App/` | SwiftUI macOS app (XcodeGen `project.yml`) |
| `backend/` | Python orchestrator + engine adapters (uv project) |
| `scripts/` | Build, bundle-Python, sign, notarize, DMG packaging |
| `docs/` | Design notes |

## Development

### Backend
```bash
cd backend
uv sync
uv run pytest
uv run python -m tinyforge --host 127.0.0.1 --port 0   # prints a JSON "ready" line with the bound port
```

### App
```bash
cd App
xcodegen generate
open TinyForge.xcodeproj
```

## Status

Milestone **M0 — foundations / walking skeleton** in progress. See `docs/` and the milestone roadmap.
