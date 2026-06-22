"""CLI entry point: `python -m tinyforge` / the `tinyforge` console script."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from tinyforge.server import run


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="tinyforge",
        description="TinyForge ML backend (FastAPI sidecar).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind host (loopback only)")
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="bind port (0 = OS-chosen ephemeral port; the chosen port is printed)",
    )
    args = parser.parse_args(argv)
    run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
