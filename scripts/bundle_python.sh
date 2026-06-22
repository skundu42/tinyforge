#!/usr/bin/env bash
# Build a relocatable, self-contained Python runtime with the backend's locked
# dependencies, for embedding at TinyForge.app/Contents/Resources/python.
#
# Uses a uv-managed python-build-standalone interpreter (relocatable: it resolves
# sys.prefix relative to its own executable), with deps installed directly into
# its site-packages (no venv indirection). The app invokes it as
# `python/bin/python3 -m tinyforge`, so console-script shebangs don't matter.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND="$ROOT/backend"
OUT="${1:-$ROOT/build/python-runtime}"
PYVER="${TINYFORGE_PYVER:-3.13}"

echo "==> ensuring uv-managed CPython $PYVER"
uv python install "$PYVER"
PY="$(uv python find "$PYVER")"
PYROOT="$(cd "$(dirname "$PY")/.." && pwd)"
echo "    interpreter: $PY"
echo "    root:        $PYROOT"

echo "==> staging interpreter into $OUT/python"
rm -rf "$OUT"
mkdir -p "$OUT"
cp -R "$PYROOT/" "$OUT/python/"
BPY="$OUT/python/bin/python3"
# The copied interpreter carries uv's PEP-668 "externally managed" marker;
# remove it so we can install dependencies directly into its site-packages.
rm -f "$OUT/python/lib/python3."*"/EXTERNALLY-MANAGED"

echo "==> exporting locked requirements"
( cd "$BACKEND" && uv export --no-hashes --no-emit-project --no-dev > "$OUT/requirements.txt" )

echo "==> installing dependencies into the bundled interpreter (large: torch/mlx/transformers)"
uv pip install --python "$BPY" -r "$OUT/requirements.txt"

echo "==> installing the tinyforge package"
uv pip install --python "$BPY" --no-deps "$BACKEND"

echo "==> precompiling + pruning caches"
"$BPY" -m compileall -q "$OUT/python/lib" >/dev/null 2>&1 || true
find "$OUT/python" -name "__pycache__" -prune -false -o -name "*.pyc" -delete 2>/dev/null || true

echo "==> smoke-testing the bundled runtime"
"$BPY" -c "import tinyforge, fastapi, uvicorn, mlx, mlx_lm; print('bundled runtime OK:', tinyforge.__version__)"

echo "==> done -> $OUT/python ($(du -sh "$OUT/python" | cut -f1))"
