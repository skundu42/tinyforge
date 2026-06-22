# TinyForge roadmap

Milestones are each independently shippable and verifiable. M0–M5 build the LLM
finetuning product; M6 adds breadth; M7 is packaging/polish.

| Milestone | Scope | Status |
|-----------|-------|--------|
| **M0** | Foundations / walking skeleton: SwiftUI app ⇄ embedded Python FastAPI sidecar, process lifecycle, health/runtime | ✅ done |
| M1 | HuggingFace Hub browse/download + cache + token auth | ⬜ next |
| M2 | Dataset builder (import, format, tokenize preview, splits) | ⬜ |
| M3 | LLM LoRA finetuning (MLX) + live dashboards + experiment tracking | ⬜ |
| M4 | Inference playground (base vs finetuned, streaming) | ⬜ |
| M5 | Exports (safetensors/LoRA, GGUF, Core ML, MLX) + push to Hub | ⬜ |
| M6 | PyTorch/MPS engine: vision, audio, from-scratch, TRL methods | ⬜ |
| M7 | Production hardening: notarized DMG, native MLX-Swift inference, auto-update | ⬜ |

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

## Conventions
- TDD: tests first (see `backend/tests/`, `App/Tests/`).
- Backend deps pinned via `uv.lock`; the Xcode project is generated from `App/project.yml` (`xcodegen generate`).
- Loopback-only API + per-launch token; training/export jobs will run as isolated worker subprocesses (M3+).
