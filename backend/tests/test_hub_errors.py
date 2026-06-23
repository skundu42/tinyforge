"""Tests for shared HuggingFace error classification (gated / not-found)."""

from tinyforge.hub.errors import classify_hub_error


def _named_error(name: str) -> Exception:
    return type(name, (Exception,), {})("boom")


class _Resp:
    def __init__(self, status: int) -> None:
        self.status_code = status


def test_classifies_gated_by_class_name() -> None:
    status, message = classify_hub_error(_named_error("GatedRepoError"))
    assert status == 403
    assert "gated" in message.lower()


def test_classifies_not_found_by_class_name() -> None:
    status, message = classify_hub_error(_named_error("RepositoryNotFoundError"))
    assert status == 404
    assert "not found" in message.lower()


def test_classifies_by_response_status() -> None:
    exc = Exception("x")
    exc.response = _Resp(403)  # type: ignore[attr-defined]
    assert classify_hub_error(exc)[0] == 403


def test_returns_none_for_unrelated_error() -> None:
    assert classify_hub_error(ValueError("boom")) is None
