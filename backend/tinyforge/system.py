"""Runtime introspection: Python/platform info and ML engine availability.

The app surfaces this so the UI can show what the bundled backend can do
(which engines are importable) without launching a job.
"""

from __future__ import annotations

import importlib.util
import platform
import sys

# Optional ML backends we probe for. Presence is checked lazily via import
# spec lookup so this stays cheap and never imports heavy native libraries.
_ENGINE_MODULES = ("mlx", "mlx_lm", "mlx_vlm", "torch", "transformers")


def engine_availability() -> dict[str, bool]:
    """Return a map of engine module name -> importable (without importing it)."""
    return {name: importlib.util.find_spec(name) is not None for name in _ENGINE_MODULES}


def runtime_info() -> dict[str, object]:
    """Return Python/platform details plus engine availability."""
    return {
        "python_version": platform.python_version(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "engines": engine_availability(),
    }
