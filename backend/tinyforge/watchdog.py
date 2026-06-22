"""Parent-death watchdog.

macOS has no PR_SET_PDEATHSIG, so if the host app force-quits or crashes the
backend would reparent to launchd and keep holding GPU/memory. This polls the
parent PID and self-terminates when it changes (the parent is gone).
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable


def parent_died(initial_ppid: int, current_ppid: int) -> bool:
    """True if the parent process is gone (its PID changed; on macOS → 1)."""
    return current_ppid != initial_ppid


def _default_on_death() -> None:  # pragma: no cover - exercised only in production
    os._exit(0)


def start_parent_death_watchdog(
    initial_ppid: int | None = None,
    get_ppid: Callable[[], int] = os.getppid,
    on_death: Callable[[], None] = _default_on_death,
    interval: float = 1.0,
) -> threading.Thread:
    """Start a daemon thread that calls `on_death` once the parent disappears."""
    base = get_ppid() if initial_ppid is None else initial_ppid

    def loop() -> None:
        while True:
            time.sleep(interval)
            if parent_died(base, get_ppid()):
                on_death()
                return

    thread = threading.Thread(target=loop, name="parent-death-watchdog", daemon=True)
    thread.start()
    return thread
