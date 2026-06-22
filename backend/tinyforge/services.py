"""Application service container wired into the FastAPI app.

Holds the long-lived services the routes depend on. `build_services()` wires the
real implementations together (the Hub client takes its token from auth so login
state is always current); tests inject fakes instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Services:
    auth: Any
    hub: Any
    downloads: Any
    cache: Any
    datasets: Any


def build_services() -> Services:
    from tinyforge.datasets.registry import DatasetRegistry
    from tinyforge.datasets.service import DatasetService
    from tinyforge.hub.auth import AuthService
    from tinyforge.hub.cache import CacheService
    from tinyforge.hub.client import HubClient
    from tinyforge.hub.downloads import DownloadManager
    from tinyforge.paths import datasets_dir

    auth = AuthService()
    return Services(
        auth=auth,
        hub=HubClient(token_provider=auth.effective_token),
        downloads=DownloadManager(),
        cache=CacheService(),
        datasets=DatasetService(DatasetRegistry(datasets_dir())),
    )
