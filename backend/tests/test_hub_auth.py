"""Tests for HuggingFace auth: token resolution + AuthService."""

from tinyforge.hub.auth import AuthService, resolve_token


def test_resolve_token_prefers_env_over_stored() -> None:
    assert resolve_token({"HF_TOKEN": "envtok"}, "stored") == "envtok"


def test_resolve_token_falls_back_to_stored() -> None:
    assert resolve_token({}, "stored") == "stored"


def test_resolve_token_is_none_when_absent() -> None:
    assert resolve_token({}, None) is None


def test_resolve_token_ignores_blank_env() -> None:
    assert resolve_token({"HF_TOKEN": "   "}, "stored") == "stored"


def test_status_logged_out_when_no_token() -> None:
    service = AuthService(
        env={},
        get_token_fn=lambda: None,
        whoami_fn=lambda token=None: {"name": "should-not-be-called"},
    )
    status = service.status()
    assert status.logged_in is False
    assert status.name is None


def test_status_logged_in_uses_whoami_name() -> None:
    service = AuthService(
        env={},
        get_token_fn=lambda: "tok",
        whoami_fn=lambda token=None: {"name": "alice"},
    )
    status = service.status()
    assert status.logged_in is True
    assert status.name == "alice"


def test_status_treats_invalid_token_as_logged_out() -> None:
    def boom(token=None):
        raise ValueError("invalid token")

    service = AuthService(env={}, get_token_fn=lambda: "bad", whoami_fn=boom)
    assert service.status().logged_in is False


def test_login_calls_login_fn_and_returns_status() -> None:
    captured: dict[str, str] = {}

    def fake_login(token, **_):
        captured["token"] = token

    service = AuthService(
        env={},
        login_fn=fake_login,
        get_token_fn=lambda: "tok",
        whoami_fn=lambda token=None: {"name": "bob"},
    )
    status = service.login("secret")
    assert captured["token"] == "secret"
    assert status.logged_in is True
    assert status.name == "bob"
