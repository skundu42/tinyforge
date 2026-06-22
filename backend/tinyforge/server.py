"""Server bootstrap: bind an ephemeral loopback port, announce it, run uvicorn.

Protocol contract with the Swift host:
  * The backend binds 127.0.0.1 on an OS-chosen port.
  * The FIRST line written to stdout is a JSON "ready" object: {"event":"ready","port":<int>}.
  * All logs go to stderr, so stdout stays a clean control channel.
  * The token is supplied via the TINYFORGE_TOKEN env var by the host. When run
    standalone (dev), a token is generated and echoed in the ready line.
"""

from __future__ import annotations

import copy
import json
import secrets
import socket
import sys
from collections.abc import Mapping
from typing import TextIO

from tinyforge.api.app import create_app

TOKEN_ENV = "TINYFORGE_TOKEN"


def resolve_token(env: Mapping[str, str]) -> tuple[str, bool]:
    """Return (token, was_generated). Uses the env var if non-empty, else generates."""
    existing = env.get(TOKEN_ENV, "").strip()
    if existing:
        return existing, False
    return secrets.token_urlsafe(32), True


def bind_localhost_socket(port: int = 0) -> tuple[socket.socket, int]:
    """Bind a TCP socket to 127.0.0.1:<port> (0 = OS picks). Returns (sock, actual_port)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    actual_port = sock.getsockname()[1]
    return sock, actual_port


def build_ready_line(port: int, token: str, token_was_generated: bool) -> str:
    """Serialize the one-line ready announcement. Only echoes the token in dev."""
    payload: dict[str, object] = {"event": "ready", "port": port}
    if token_was_generated:
        payload["token"] = token
    return json.dumps(payload)


def _emit_ready(port: int, token: str, token_was_generated: bool, stream: TextIO) -> None:
    stream.write(build_ready_line(port, token, token_was_generated) + "\n")
    stream.flush()


def _stderr_log_config() -> dict:
    """Uvicorn's default logging, redirected entirely to stderr."""
    import uvicorn.config

    config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    config["handlers"]["default"]["stream"] = "ext://sys.stderr"
    config["handlers"]["access"]["stream"] = "ext://sys.stderr"
    return config


def run(host: str = "127.0.0.1", port: int = 0, env: Mapping[str, str] | None = None) -> None:
    """Bind, announce readiness on stdout, then serve the API on the bound socket."""
    import os

    import uvicorn

    env = os.environ if env is None else env
    token, generated = resolve_token(env)
    sock, actual_port = bind_localhost_socket(port if host == "127.0.0.1" else 0)

    app = create_app(token=token)
    _emit_ready(actual_port, token, generated, sys.stdout)

    # Self-terminate if the host app dies (no PR_SET_PDEATHSIG on macOS).
    from tinyforge.watchdog import start_parent_death_watchdog

    start_parent_death_watchdog()

    server = uvicorn.Server(uvicorn.Config(app, log_config=_stderr_log_config()))
    server.run(sockets=[sock])
