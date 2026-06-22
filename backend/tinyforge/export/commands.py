"""Build mlx_lm fuse/convert command lines for exports."""

from __future__ import annotations


def build_fuse_command(
    python_exe: str,
    base_repo: str,
    adapter_path: str,
    save_path: str,
    gguf_path: str | None = None,
) -> list[str]:
    command = [
        python_exe, "-m", "mlx_lm", "fuse",
        "--model", base_repo,
        "--adapter-path", adapter_path,
        "--save-path", save_path,
    ]
    if gguf_path:
        command += ["--gguf-path", gguf_path]
    return command


def build_convert_command(
    python_exe: str, hf_path: str, mlx_path: str, q_bits: int = 4
) -> list[str]:
    return [
        python_exe, "-m", "mlx_lm", "convert",
        "--hf-path", hf_path,
        "--mlx-path", mlx_path,
        "-q",
        "--q-bits", str(q_bits),
    ]
