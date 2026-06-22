"""Tests for the parent-death watchdog (prevents orphaned backends)."""

import threading

from tinyforge import watchdog


def test_parent_died_detects_ppid_change() -> None:
    # On macOS, a dead parent reparents the child to launchd (pid 1).
    assert watchdog.parent_died(initial_ppid=1000, current_ppid=1) is True
    assert watchdog.parent_died(initial_ppid=1000, current_ppid=1000) is False


def test_watchdog_invokes_on_death_when_parent_changes() -> None:
    died = threading.Event()
    watchdog.start_parent_death_watchdog(
        initial_ppid=1000,
        get_ppid=lambda: 1,  # simulate the parent already gone
        on_death=died.set,
        interval=0.01,
    )
    assert died.wait(timeout=2.0) is True


def test_watchdog_stays_quiet_while_parent_alive() -> None:
    died = threading.Event()
    watchdog.start_parent_death_watchdog(
        initial_ppid=1000,
        get_ppid=lambda: 1000,  # parent unchanged
        on_death=died.set,
        interval=0.01,
    )
    assert died.wait(timeout=0.2) is False
