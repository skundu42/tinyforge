#!/usr/bin/env bash
# End-to-end release build: compile the app (Release), embed the bundled Python
# runtime, sign inside-out with Developer ID, and produce a signed DMG.
# Notarization is a separate step (scripts/notarize.sh) as it needs your Apple
# ID credentials.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
RUNTIME="$BUILD/python-runtime/python"
IDENTITY="${SIGN_IDENTITY:-Developer ID Application: Sandipan Kundu (K9ATTR44A7)}"

echo "==> [1/5] bundle Python runtime (if missing)"
[[ -x "$RUNTIME/bin/python3" ]] || "$ROOT/scripts/bundle_python.sh"

echo "==> [2/5] build the app (Release)"
( cd "$ROOT/App" && xcodegen generate >/dev/null && \
  xcodebuild build -project TinyForge.xcodeproj -scheme TinyForge \
    -configuration Release -derivedDataPath "$BUILD" -skipMacroValidation \
    CODE_SIGN_IDENTITY="-" CODE_SIGNING_REQUIRED=NO >/dev/null )
APP="$BUILD/Build/Products/Release/TinyForge.app"

echo "==> [3/5] embed the Python runtime"
rm -rf "$APP/Contents/Resources/python"
ditto "$RUNTIME" "$APP/Contents/Resources/python"

echo "==> [4/5] sign inside-out (Developer ID)"
"$ROOT/scripts/sign.sh" "$APP" "$IDENTITY"

echo "==> [5/5] build signed DMG"
"$ROOT/scripts/package_dmg.sh" "$APP" "$BUILD/TinyForge.dmg" "$IDENTITY"

echo ""
echo "Release app: $APP"
echo "DMG:         $BUILD/TinyForge.dmg"
echo "Next:        scripts/notarize.sh \"$BUILD/TinyForge.dmg\""
