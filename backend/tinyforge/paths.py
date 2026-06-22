"""Application data directories."""

from __future__ import annotations

import os
from pathlib import Path


def app_data_dir() -> Path:
    override = os.environ.get("TINYFORGE_DATA_DIR")
    base = Path(override) if override else Path.home() / "Library" / "Application Support" / "TinyForge"
    base.mkdir(parents=True, exist_ok=True)
    return base


def datasets_dir() -> Path:
    path = app_data_dir() / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path
