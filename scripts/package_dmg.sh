#!/usr/bin/env bash
# Build a signed, compressed DMG containing TinyForge.app + an /Applications link.
set -euo pipefail

APP="$1"
DMG="${2:-build/TinyForge.dmg}"
IDENTITY="${3:-Developer ID Application: Sandipan Kundu (K9ATTR44A7)}"

STAGING="$(mktemp -d)"
# `ditto` preserves the symlinked Python tree correctly (zip/cp -R can mangle it).
ditto "$APP" "$STAGING/$(basename "$APP")"
ln -s /Applications "$STAGING/Applications"

mkdir -p "$(dirname "$DMG")"
rm -f "$DMG"
hdiutil create -volname "TinyForge" -srcfolder "$STAGING" -ov -format UDZO "$DMG"
codesign --force --timestamp --sign "$IDENTITY" "$DMG"
rm -rf "$STAGING"

echo "==> DMG: $DMG ($(du -sh "$DMG" | cut -f1))"
