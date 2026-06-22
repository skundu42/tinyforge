#!/usr/bin/env bash
# Code-sign TinyForge.app inside-out (NOT `codesign --deep`): every nested
# Mach-O in the bundled Python is signed first (deepest), then the interpreter
# executables with hardened-runtime entitlements, then the app last.
set -euo pipefail

APP="$1"
IDENTITY="${2:-Developer ID Application: Sandipan Kundu (K9ATTR44A7)}"
ENTITLEMENTS="$(cd "$(dirname "$0")" && pwd)/TinyForge.entitlements"
PYDIR="$APP/Contents/Resources/python"

# Ad-hoc signing (identity "-", e.g. an unsigned CI build) can't reach Apple's
# secure timestamp server, so request no timestamp in that case.
TIMESTAMP="--timestamp"
[[ "$IDENTITY" == "-" ]] && TIMESTAMP="--timestamp=none"

sign_lib() { codesign --force --options runtime "$TIMESTAMP" --sign "$IDENTITY" "$1"; }
sign_exe() { codesign --force --options runtime "$TIMESTAMP" --entitlements "$ENTITLEMENTS" --sign "$IDENTITY" "$1"; }

if [[ -d "$PYDIR" ]]; then
  echo "==> signing nested native libraries (.so/.dylib)"
  find "$PYDIR" -type f \( -name "*.so" -o -name "*.dylib" \) -print0 | while IFS= read -r -d '' lib; do
    sign_lib "$lib"
  done

  echo "==> signing interpreter executables (with entitlements)"
  find "$PYDIR/bin" -type f | while IFS= read -r f; do
    if file "$f" | grep -q "Mach-O"; then sign_exe "$f"; fi
  done
  # Any other Mach-O executables (e.g. dylibs without extension, console scripts).
  find "$PYDIR" -type f -perm +111 ! -name "*.so" ! -name "*.dylib" | while IFS= read -r f; do
    if file "$f" | grep -q "Mach-O executable"; then sign_exe "$f"; fi
  done
fi

echo "==> signing the app bundle"
codesign --force --options runtime "$TIMESTAMP" --entitlements "$ENTITLEMENTS" --sign "$IDENTITY" "$APP"

echo "==> verifying signature"
codesign --verify --deep --strict --verbose=2 "$APP"
echo "==> gatekeeper assessment (expected to fail until notarized — informational)"
spctl -a -vvv -t exec "$APP" 2>&1 || true
echo "==> signed: $APP"
