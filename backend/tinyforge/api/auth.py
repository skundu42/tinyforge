"""Per-launch bearer-token auth.

The Swift app generates a random token, passes it to the backend on spawn, and
sends it on every request. The backend binds to 127.0.0.1 only, so the token is
defence-in-depth against other local processes hitting the API.
"""

from __future__ import annotations

import hmac
from collections.abc import Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)


def make_token_dependency(token: str) -> Callable[..., None]:
    """Build a FastAPI dependency that requires `Authorization: Bearer <token>`."""

    def require_token(
        creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ) -> None:
        supplied = creds.credentials if creds else ""
        # Constant-time comparison to avoid leaking the token via timing.
        if not hmac.compare_digest(supplied, token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid or missing token",
            )

    return require_token
