# CI/CD — release DMG on tag

`.github/workflows/release.yml` builds the macOS `.dmg` and attaches it to a
GitHub Release whenever a version tag is pushed.

## Trigger a release

```bash
# tag a commit and push the tag
git tag v0.1.0
git push origin v0.1.0
```

The workflow runs on an Apple-Silicon `macos-15` runner and:

1. installs `xcodegen` + `uv`, selects Xcode, and downloads the Metal Toolchain
   (needed for MLX-Swift's shaders);
2. runs `scripts/build_release.sh` — bundles the Python runtime, builds the app
   (Release), embeds the runtime, signs it, and packages `TinyForge-<tag>.dmg`;
3. (if signing secrets are set) notarizes + staples the DMG;
4. uploads the DMG as a workflow artifact **and** attaches it to the
   **GitHub Release** for the tag (with auto-generated notes).

You can also run it manually from the **Actions** tab (**Run workflow** →
enter a tag) for a dry run — that builds the artifact but does not publish a
Release.

> The runner must provide **Xcode 26+ (Swift 6.3)**. `latest-stable` is used; pin
> a specific version in the *Select Xcode* step if needed.

## Signing & notarization (optional)

Without secrets, the job still produces a working **ad-hoc (unsigned)** DMG —
users must right-click → **Open** the first time. To ship a **Developer-ID
signed + notarized** DMG, add these repository secrets
(**Settings → Secrets and variables → Actions**):

| Secret | What it is | How to get it |
|--------|------------|---------------|
| `DEVELOPER_ID_APPLICATION` | The identity string, e.g. `Developer ID Application: Your Name (TEAMID)` | `security find-identity -v -p codesigning` |
| `DEVELOPER_ID_CERT_P12` | Base64 of your exported Developer ID cert | Keychain Access → export the cert + key as `.p12`, then `base64 -i cert.p12 \| pbcopy` |
| `DEVELOPER_ID_CERT_PASSWORD` | Password you set when exporting the `.p12` | — |
| `APPLE_ID` | Apple ID email used for notarization | your developer-account email |
| `APPLE_APP_PASSWORD` | App-specific password for that Apple ID | [appleid.apple.com](https://appleid.apple.com) → Sign-In & Security → App-Specific Passwords |
| `APPLE_TEAM_ID` | Your 10-char Team ID | Apple Developer → Membership |

When `DEVELOPER_ID_APPLICATION` is present the workflow imports the cert into a
temporary keychain, builds with `SIGN_IDENTITY` set to it, then notarizes with
the Apple-ID credentials. When it's absent, the build uses the ad-hoc identity
`-` and skips notarization.

## Notes

- The bundled Python runtime (~1 GB) is **cached** on `uv.lock` /
  `pyproject.toml` / `bundle_python.sh`, so most runs skip the slow re-bundle.
- The same `scripts/*.sh` drive local releases and CI — `SIGN_IDENTITY=-`
  builds an ad-hoc DMG locally too.
- Local notarization (keychain profile) is described in
  [`docs/packaging.md`](packaging.md).
