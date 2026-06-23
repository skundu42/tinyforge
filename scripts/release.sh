#!/usr/bin/env bash
# Cut a release: bump the in-repo version in every place it lives, refresh the
# lockfile, commit, create an annotated `vX.Y.Z` tag, and push. The pushed tag
# triggers .github/workflows/release.yml, which builds + publishes the DMG.
#
# This is the ONLY supported way to release: release.yml re-checks that the
# in-repo versions match the tag and fails the build otherwise, so versions can
# never drift out of sync with a release.
#
# Usage: scripts/release.sh X.Y.Z      (e.g. scripts/release.sh 0.1.2)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

version="${1:-}"
if ! printf '%s' "$version" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
  echo "usage: scripts/release.sh X.Y.Z   (e.g. scripts/release.sh 0.1.2)" >&2
  exit 2
fi
tag="v$version"

# Refuse to run on a dirty tree so the release commit is exactly the version bump.
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "error: working tree has uncommitted changes — commit or stash them first." >&2
  exit 1
fi
if git rev-parse "$tag" >/dev/null 2>&1; then
  echo "error: tag $tag already exists." >&2
  exit 1
fi

echo "==> bumping version to $version"
sed -i.bak -E "s/^version = \".*\"/version = \"$version\"/" backend/pyproject.toml && rm backend/pyproject.toml.bak
sed -i.bak -E "s/(MARKETING_VERSION: )\".*\"/\1\"$version\"/" App/project.yml && rm App/project.yml.bak
sed -i.bak -E "s/^__version__ = \".*\"/__version__ = \"$version\"/" backend/tinyforge/__init__.py && rm backend/tinyforge/__init__.py.bak

echo "==> refreshing backend/uv.lock"
( cd backend && uv lock )

echo "==> committing + tagging $tag"
git add backend/pyproject.toml App/project.yml backend/tinyforge/__init__.py backend/uv.lock
git commit -m "release $tag"
git tag -a "$tag" -m "TinyForge $tag"

branch="$(git rev-parse --abbrev-ref HEAD)"
echo "==> pushing $branch + $tag"
git push origin "$branch" --follow-tags

echo "Done — release.yml will build and publish the DMG for $tag."
