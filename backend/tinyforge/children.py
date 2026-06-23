"""Registry of live child processes so they can be reaped when the backend dies.

Training (`mlx_lm`) and export (`mlx_lm fuse/convert`) run as child processes.
macOS has no PR_SET_PDEATHSIG, so when the backend exits — gracefully on SIGTERM,
or abruptly when the watchdog fires `os._exit` after the host app is gone — those
children would otherwise reparent to launchd and keep holding the GPU. Children
are spawned in their own session (`start_new_session=True`) so each one's PID is
also its process-group id; killing that group reaps the child and any workers it
spawned.
"""

from __future__ import annotations

import os
import signal
import threading
from collections.abc import Callable


def _killpg(pid: int, sig: int) -> None:
    os.killpg(pid, sig)


class ChildRegistry:
    """Thread-safe set of live child PIDs (each its own process-group leader)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pids: set[int] = set()

    def register(self, pid: int) -> None:
        with self._lock:
            self._pids.add(pid)

    def unregister(self, pid: int) -> None:
        with self._lock:
            self._pids.discard(pid)

    def terminate_all(
        self, kill: Callable[[int, int], None] = _killpg, sig: int = signal.SIGTERM
    ) -> None:
        """Signal every live child's process group. Safe to call from a signal
        handler or just before `os._exit`; children that already exited are
        ignored."""
        with self._lock:
            pids = list(self._pids)
        for pid in pids:
            try:
                kill(pid, sig)
            except (ProcessLookupError, OSError):
                pass


# Process-wide singleton wired into the runner, exporter, watchdog, and the
# app's graceful-shutdown hook.
child_registry = ChildRegistry()
