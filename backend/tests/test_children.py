"""Tests for the live child-process registry used to reap orphans on death."""

import signal

from tinyforge.children import ChildRegistry


def test_terminate_all_signals_each_registered_child() -> None:
    killed: list[tuple[int, int]] = []
    registry = ChildRegistry()
    registry.register(101)
    registry.register(202)

    registry.terminate_all(kill=lambda pid, sig: killed.append((pid, sig)))

    assert {pid for pid, _ in killed} == {101, 202}
    assert all(sig == signal.SIGTERM for _, sig in killed)


def test_unregister_removes_child() -> None:
    killed: list[int] = []
    registry = ChildRegistry()
    registry.register(303)
    registry.unregister(303)

    registry.terminate_all(kill=lambda pid, sig: killed.append(pid))

    assert killed == []


def test_terminate_all_ignores_already_dead_children() -> None:
    registry = ChildRegistry()
    registry.register(404)

    def kill(_pid: int, _sig: int) -> None:
        raise ProcessLookupError

    # A child that already exited must not crash the sweep.
    registry.terminate_all(kill=kill)
