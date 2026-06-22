"""Test the CLI entry point dispatches host/port into server.run."""

from tinyforge import __main__


def test_main_passes_host_and_port_to_run(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(host: str, port: int, **_: object) -> None:
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(__main__, "run", fake_run)
    __main__.main(["--host", "127.0.0.1", "--port", "0"])

    assert captured == {"host": "127.0.0.1", "port": 0}
