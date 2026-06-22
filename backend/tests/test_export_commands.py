"""Tests for export command builders."""

from tinyforge.export.commands import build_convert_command, build_fuse_command


def test_fuse_command_basic() -> None:
    cmd = build_fuse_command("py", base_repo="base/m", adapter_path="/a", save_path="/out/fused")
    assert cmd[:4] == ["py", "-m", "mlx_lm", "fuse"]
    joined = " ".join(cmd)
    assert "--model base/m" in joined
    assert "--adapter-path /a" in joined
    assert "--save-path /out/fused" in joined
    assert "--gguf-path" not in joined


def test_fuse_command_with_gguf() -> None:
    cmd = build_fuse_command("py", "base/m", "/a", "/out/fused", gguf_path="/out/model.gguf")
    assert "--gguf-path /out/model.gguf" in " ".join(cmd)


def test_convert_command_quantizes() -> None:
    cmd = build_convert_command("py", hf_path="/out/fused", mlx_path="/out/mlx", q_bits=4)
    assert cmd[:4] == ["py", "-m", "mlx_lm", "convert"]
    joined = " ".join(cmd)
    assert "--hf-path /out/fused" in joined
    assert "--mlx-path /out/mlx" in joined
    assert "-q" in cmd
    assert "--q-bits 4" in joined
