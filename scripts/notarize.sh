#!/usr/bin/env bash
# Notarize + staple an app or DMG.
#
# One-time setup (stores an app-specific password in the keychain):
#   xcrun notarytool store-credentials TinyForgeNotary \
#     --apple-id "you@example.com" --team-id K9ATTR44A7 --password "<app-specific-pw>"
set -euo pipefail

TARGET="$1"                         # TinyForge.app or TinyForge.dmg
PROFILE="${2:-TinyForgeNotary}"

if [[ "$TARGET" == *.app ]]; then
  SUBMIT="/tmp/tinyforge-notarize.zip"
  ditto -c -k --keepParent "$TARGET" "$SUBMIT"   # ditto, never zip (symlinks)
else
  SUBMIT="$TARGET"
fi

echo "==> submitting to Apple notary service"
xcrun notarytool submit "$SUBMIT" --keychain-profile "$PROFILE" --wait
# On failure: xcrun notarytool log <submission-id> --keychain-profile "$PROFILE"

echo "==> stapling ticket"
xcrun stapler staple "$TARGET"
xcrun stapler validate "$TARGET"
echo "==> notarized + stapled: $TARGET"
