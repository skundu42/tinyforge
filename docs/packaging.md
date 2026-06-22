# Packaging TinyForge as a notarized DMG

TinyForge ships as a self-contained `.app`: a relocatable Python runtime (with
torch, mlx, transformers, etc.) is bundled at `Contents/Resources/python`, and
the app launches it as `python/bin/python3 -m tinyforge` — no developer
toolchain required on the user's machine.

## One command

```bash
scripts/build_release.sh        # bundle python → build app → embed → sign → DMG
scripts/notarize.sh build/TinyForge.dmg
```

## Steps

1. **`scripts/bundle_python.sh`** — copies a uv-managed python-build-standalone
   CPython (relocatable: resolves `sys.prefix` from its own executable) into a
   staging dir and installs the backend's locked dependencies directly into its
   site-packages. ~1 GB. The app invokes it via `-m tinyforge`, so console-script
   shebangs are irrelevant; the PEP-668 marker is removed so deps can install.

2. **Build (Release)** + **embed** — `xcodebuild -configuration Release`, then
   `ditto` the runtime into `Contents/Resources/python`.

3. **`scripts/sign.sh`** — signs **inside-out** (NOT `codesign --deep`): every
   nested `.so`/`.dylib` first, then the interpreter executables with the
   hardened-runtime entitlements (`scripts/TinyForge.entitlements`:
   `disable-library-validation`, `allow-unsigned-executable-memory`,
   `allow-jit`), then the app last. Uses `Developer ID Application`.

4. **`scripts/package_dmg.sh`** — `ditto`s the app into a staging folder with an
   `/Applications` symlink and builds a signed UDZO DMG.

5. **`scripts/notarize.sh`** — submits to Apple's notary service and staples.
   One-time credential setup:
   ```bash
   xcrun notarytool store-credentials TinyForgeNotary \
     --apple-id "you@example.com" --team-id K9ATTR44A7 --password "<app-specific-pw>"
   ```

## Verified

- The bundled runtime starts the backend (emits the ready line) standalone.
- A Release app embedding it spawns the **bundled** interpreter
  (`TinyForge.app/Contents/Resources/python/bin/python3`) — confirmed
  self-contained, with no dependency on the repo's dev venv.

## Notes / gotchas (from the build)

- `uv` won't `pip install` into a copied managed interpreter until its
  `EXTERNALLY-MANAGED` marker is removed (handled by `bundle_python.sh`).
- Signing every nested Mach-O with `--timestamp` makes a network call each — the
  full Developer-ID sign of the ~1 GB runtime takes a while.
- Use `ditto` (never `zip`/`cp -R`) for the symlinked Python tree and the
  notarization payload.
- Notarization needs your Apple ID app-specific password, so it's a separate
  step from `build_release.sh`.
