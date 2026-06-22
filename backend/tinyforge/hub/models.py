"""Pydantic models for the HuggingFace Hub API surface."""

from __future__ import annotations

from pydantic import BaseModel


class AuthStatus(BaseModel):
    logged_in: bool
    name: str | None = None


class HubModel(BaseModel):
    id: str
    author: str | None = None
    downloads: int | None = None
    likes: int | None = None
    gated: bool = False
    private: bool = False
    pipeline_tag: str | None = None
    library_name: str | None = None
    tags: list[str] = []
    last_modified: str | None = None


class HubDataset(BaseModel):
    id: str
    author: str | None = None
    downloads: int | None = None
    likes: int | None = None
    gated: bool = False
    private: bool = False
    tags: list[str] = []
    last_modified: str | None = None


class HubFile(BaseModel):
    filename: str
    size: int | None = None


class HubModelDetail(HubModel):
    siblings: list[HubFile] = []
    total_size: int | None = None
    readme: str | None = None


class DownloadPlanFile(BaseModel):
    filename: str
    size: int
    will_download: bool


class DownloadPlan(BaseModel):
    repo_id: str
    repo_type: str
    files: list[DownloadPlanFile]
    total_bytes: int
    already_cached_bytes: int


class DownloadProgress(BaseModel):
    id: str
    repo_id: str
    repo_type: str
    total_bytes: int
    downloaded_bytes: int
    fraction: float
    state: str  # pending | running | completed | error
    error: str | None = None
    local_path: str | None = None


class CachedRepo(BaseModel):
    repo_id: str
    repo_type: str
    size_on_disk: int
    nb_files: int
    last_accessed: float | None = None


class CacheInfo(BaseModel):
    size_on_disk: int
    repos: list[CachedRepo]
