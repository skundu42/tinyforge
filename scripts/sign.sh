#!/usr/bin/env bash
# Code-sign TinyForge.app inside-out (NOT `codesign --deep`): sign every nested
# Mach-O — dynamic libraries (.so/.dylib across the bundled Python/torch/mlx
# tree) and executables (interpreter, protoc, torch_shm_manager, console
# scripts, …) — then nested frameworks, then the app last. Binaries are
# classified by `file` output, not by extension: `file` reports e.g.
# "Mach-O 64-bit executable arm64", so matching the literal "Mach-O executable"
# misses them — that gap previously left torch/bin executables ad-hoc and failed
# notarization.
set -euo pipefail

APP="$1"
IDENTITY="${2:-Developer ID Application: Sandipan Kundu (K9ATTR44A7)}"
ENTITLEMENTS="$(cd "$(dirname "$0")" && pwd)/TinyForge.entitlements"

# Ad-hoc signing (identity "-", e.g. an unsigned CI build) can't reach Apple's
# secure timestamp server, so request no timestamp in that case.
TIMESTAMP="--timestamp"
[[ "$IDENTITY" == "-" ]] && TIMESTAMP="--timestamp=none"

sign_lib() { codesign --force --options runtime "$TIMESTAMP" --sign "$IDENTITY" "$1"; }
sign_exe() { codesign --force --options runtime "$TIMESTAMP" --entitlements "$ENTITLEMENTS" --sign "$IDENTITY" "$1"; }

# 1. Dynamic libraries by name (the bulk — .so/.dylib anywhere in the bundle).
echo "==> signing nested dynamic libraries (.so/.dylib)"
find "$APP/Contents" -type f \( -name "*.so" -o -name "*.dylib" \) -print0 | while IFS= read -r -d '' f; do
  sign_lib "$f"
done

# 2. Every other executable file that is actually Mach-O (interpreter, torch/bin
#    protoc & torch_shm_manager, oddly-named libs). Classified via `file`.
echo "==> signing other nested Mach-O binaries"
find "$APP/Contents" -type f -perm +111 ! -name "*.so" ! -name "*.dylib" -print0 | while IFS= read -r -d '' f; do
  case "$(file "$f")" in
    *Mach-O*executable*) sign_exe "$f" ;;
    *Mach-O*) sign_lib "$f" ;;
  esac
done

# 3. Nested frameworks (sealed after their contents).
echo "==> signing nested frameworks"
find "$APP/Contents" -type d -name "*.framework" -print0 | while IFS= read -r -d '' fw; do
  sign_lib "$fw"
done

# 4. The app itself, last.
echo "==> signing the app bundle"
codesign --force --options runtime "$TIMESTAMP" --entitlements "$ENTITLEMENTS" --sign "$IDENTITY" "$APP"

echo "==> verifying signature"
codesign --verify --deep --strict --verbose=2 "$APP"
echo "==> gatekeeper assessment (informational; expected to fail until notarized)"
spctl -a -vvv -t exec "$APP" 2>&1 || true
echo "==> signed: $APP"
