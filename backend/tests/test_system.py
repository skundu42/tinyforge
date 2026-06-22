"""Tests for tinyforge.system runtime/engine introspection."""

from tinyforge import system


def test_engine_availability_reports_known_engines_as_bools() -> None:
    avail = system.engine_availability()
    for name in ("mlx", "mlx_lm", "torch", "transformers"):
        assert name in avail
        assert isinstance(avail[name], bool)


def test_runtime_info_reports_platform_and_engines() -> None:
    info = system.runtime_info()
    assert info["python_version"].startswith("3.")
    assert info["platform"] == "darwin"
    assert info["machine"] in ("arm64", "x86_64")
    assert isinstance(info["engines"], dict)
