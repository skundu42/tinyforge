"""Tests for server bootstrap helpers: token resolution, socket bind, ready line."""

import json
import socket

from tinyforge import server


def test_resolve_token_uses_env_when_present() -> None:
    token, generated = server.resolve_token({"TINYFORGE_TOKEN": "abc123"})
    assert token == "abc123"
    assert generated is False


def test_resolve_token_generates_when_absent() -> None:
    token, generated = server.resolve_token({})
    assert generated is True
    assert len(token) >= 16


def test_resolve_token_generates_when_blank() -> None:
    token, generated = server.resolve_token({"TINYFORGE_TOKEN": ""})
    assert generated is True
    assert len(token) >= 16


def test_bind_localhost_socket_picks_free_port_on_loopback() -> None:
    sock, port = server.bind_localhost_socket(0)
    try:
        assert port > 0
        assert sock.getsockname() == ("127.0.0.1", port)
    finally:
        sock.close()


def test_ready_line_carries_port_but_not_token_when_token_provided() -> None:
    line = server.build_ready_line(port=51820, token="secret", token_was_generated=False)
    payload = json.loads(line)
    assert payload["event"] == "ready"
    assert payload["port"] == 51820
    assert "token" not in payload  # parent already knows it; never echo on the wire


def test_ready_line_includes_token_when_generated_for_dev() -> None:
    line = server.build_ready_line(port=51820, token="secret", token_was_generated=True)
    payload = json.loads(line)
    assert payload["token"] == "secret"
