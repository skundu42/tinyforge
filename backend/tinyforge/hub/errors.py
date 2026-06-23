"""Translate HuggingFace Hub failures into actionable HTTP errors.

A gated/private repo with no token, and a mistyped repo id, are the two most
common failures across browsing, detail, and download. Classifying by class name
+ HTTP status (rather than importing huggingface_hub's error types) keeps this
testable with fakes and resilient across library versions.
"""

from __future__ import annotations

_GATED = (
    "This model is gated. Request access on its Hugging Face page, then sign in "
    "with a Hugging Face token in Settings to access it."
)
_NOT_FOUND = (
    "Model not found on Hugging Face. Check the repository id — if it is private, "
    "sign in with a Hugging Face token in Settings."
)


def classify_hub_error(exc: Exception) -> tuple[int, str] | None:
    """Return (http_status, message) for a known Hub failure, else None.

    Gated is checked before not-found because `GatedRepoError` subclasses
    `RepositoryNotFoundError`.
    """
    class_names = {cls.__name__ for cls in type(exc).__mro__}
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if "GatedRepoError" in class_names or status in (401, 403):
        return 403, _GATED
    if "RepositoryNotFoundError" in class_names or status == 404:
        return 404, _NOT_FOUND
    return None
