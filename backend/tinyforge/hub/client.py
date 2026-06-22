"""HuggingFace Hub search & detail, mapping HfApi results to typed models.

HfApi is injectable so the mapping logic is testable without network. The
effective token is supplied per-call via a provider so auth changes are picked
up without rebuilding the client.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from tinyforge.hub.models import HubDataset, HubFile, HubModel, HubModelDetail

ModelSort = str  # one of: created_at, downloads, last_modified, likes, trending_score


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _default_readme(repo_id: str, token: str | None = None) -> str | None:
    import huggingface_hub

    path = huggingface_hub.hf_hub_download(repo_id, "README.md", token=token)
    with open(path, encoding="utf-8") as handle:
        return handle.read()


class HubClient:
    def __init__(
        self,
        api: Any = None,
        *,
        token_provider: Callable[[], str | None] = lambda: None,
        readme_fn: Callable[..., str | None] = _default_readme,
    ) -> None:
        if api is None:
            import huggingface_hub

            api = huggingface_hub.HfApi()
        self._api = api
        self._token_provider = token_provider
        self._readme_fn = readme_fn

    def search_models(
        self,
        query: str | None = None,
        *,
        sort: ModelSort = "downloads",
        limit: int = 30,
        author: str | None = None,
        pipeline_tag: str | None = None,
        filter: str | None = None,
        gated: bool | None = None,
    ) -> list[HubModel]:
        results = self._api.list_models(
            search=query,
            sort=sort,
            limit=limit,
            author=author,
            pipeline_tag=pipeline_tag,
            filter=filter,
            gated=gated,
            token=self._token_provider(),
        )
        return [self._to_model(item) for item in results]

    def search_datasets(
        self,
        query: str | None = None,
        *,
        sort: str = "downloads",
        limit: int = 30,
        author: str | None = None,
    ) -> list[HubDataset]:
        results = self._api.list_datasets(
            search=query, sort=sort, limit=limit, author=author,
            token=self._token_provider(),
        )
        return [self._to_dataset(item) for item in results]

    def model_detail(self, repo_id: str) -> HubModelDetail:
        token = self._token_provider()
        info = self._api.model_info(repo_id, files_metadata=True, token=token)
        siblings = [
            HubFile(filename=s.rfilename, size=getattr(s, "size", None))
            for s in (getattr(info, "siblings", None) or [])
        ]
        sizes = [f.size for f in siblings if f.size is not None]
        total = sum(sizes) if sizes else None
        readme = self._safe_readme(repo_id, token)
        base = self._to_model(info)
        return HubModelDetail(
            **base.model_dump(), siblings=siblings, total_size=total, readme=readme
        )

    def _safe_readme(self, repo_id: str, token: str | None) -> str | None:
        try:
            return self._readme_fn(repo_id, token=token)
        except Exception:
            return None

    def _to_model(self, item: Any) -> HubModel:
        return HubModel(
            id=item.id,
            author=self._author(item),
            downloads=getattr(item, "downloads", None),
            likes=getattr(item, "likes", None),
            gated=bool(getattr(item, "gated", False)),
            private=bool(getattr(item, "private", False)),
            pipeline_tag=getattr(item, "pipeline_tag", None),
            library_name=getattr(item, "library_name", None),
            tags=list(getattr(item, "tags", None) or []),
            last_modified=_iso(getattr(item, "last_modified", None)),
        )

    def _to_dataset(self, item: Any) -> HubDataset:
        return HubDataset(
            id=item.id,
            author=self._author(item),
            downloads=getattr(item, "downloads", None),
            likes=getattr(item, "likes", None),
            gated=bool(getattr(item, "gated", False)),
            private=bool(getattr(item, "private", False)),
            tags=list(getattr(item, "tags", None) or []),
            last_modified=_iso(getattr(item, "last_modified", None)),
        )

    @staticmethod
    def _author(item: Any) -> str | None:
        author = getattr(item, "author", None)
        if author:
            return author
        return item.id.split("/")[0] if "/" in item.id else None
