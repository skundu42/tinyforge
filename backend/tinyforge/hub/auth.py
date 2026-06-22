"""HuggingFace authentication.

Token precedence matches the Hub libraries: the HF_TOKEN env var wins over the
token stored on disk. The service wraps huggingface_hub's auth functions behind
injectable callables so the logic is testable without network or global state.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping

from tinyforge.hub.models import AuthStatus


def resolve_token(env: Mapping[str, str], stored_token: str | None) -> str | None:
    """Effective token: HF_TOKEN env var takes priority over the stored token."""
    env_token = env.get("HF_TOKEN", "").strip()
    if env_token:
        return env_token
    return stored_token or None


def _default_login(token: str, **kwargs: object) -> None:
    import huggingface_hub

    huggingface_hub.login(token=token, add_to_git_credential=False)


def _default_whoami(token: str | None = None) -> dict:
    import huggingface_hub

    return huggingface_hub.whoami(token=token)


def _default_get_token() -> str | None:
    import huggingface_hub

    return huggingface_hub.get_token()


def _default_logout() -> None:
    import huggingface_hub

    huggingface_hub.logout()


class AuthService:
    def __init__(
        self,
        *,
        env: Mapping[str, str] | None = None,
        login_fn: Callable[..., None] = _default_login,
        whoami_fn: Callable[..., dict] = _default_whoami,
        get_token_fn: Callable[[], str | None] = _default_get_token,
        logout_fn: Callable[[], None] = _default_logout,
    ) -> None:
        self._env = os.environ if env is None else env
        self._login_fn = login_fn
        self._whoami_fn = whoami_fn
        self._get_token_fn = get_token_fn
        self._logout_fn = logout_fn

    def effective_token(self) -> str | None:
        return resolve_token(self._env, self._get_token_fn())

    def status(self) -> AuthStatus:
        """Best-effort login status. An invalid/expired token reads as logged out."""
        token = self.effective_token()
        if not token:
            return AuthStatus(logged_in=False)
        try:
            info = self._whoami_fn(token=token)
        except Exception:
            return AuthStatus(logged_in=False)
        return AuthStatus(logged_in=True, name=info.get("name"))

    def login(self, token: str) -> AuthStatus:
        """Persist the token and return the resulting status. Raises on failure."""
        self._login_fn(token)
        info = self._whoami_fn(token=token)
        return AuthStatus(logged_in=True, name=info.get("name"))

    def logout(self) -> None:
        self._logout_fn()
