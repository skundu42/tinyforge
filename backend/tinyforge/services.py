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
    training: Any
    inference: Any
    exports: Any


def build_services() -> Services:
    import sys

    from tinyforge.export.manager import ExportManager
    from tinyforge.export.push import push_folder
    from tinyforge.infer.service import InferenceService

    from tinyforge.datasets.registry import DatasetRegistry
    from tinyforge.datasets.service import DatasetService
    from tinyforge.hub.auth import AuthService
    from tinyforge.hub.cache import CacheService
    from tinyforge.hub.client import HubClient
    from tinyforge.hub.downloads import DownloadManager
    from tinyforge.paths import datasets_dir, exports_dir, runs_dir
    from tinyforge.train.registry import RunRegistry
    from tinyforge.train.runner import TrainingRunner
    from tinyforge.train.service import TrainingService

    auth = AuthService()
    datasets = DatasetService(DatasetRegistry(datasets_dir()))
    training = TrainingService(
        runner=TrainingRunner(python_exe=sys.executable),
        registry=RunRegistry(runs_dir()),
        runs_dir=runs_dir(),
        dataset_resolver=lambda dataset_id: datasets.get(dataset_id).path,
    )

    def resolve_run(run_id: str) -> tuple[str, str]:
        record = training.get(run_id)
        return record.config["model_repo"], record.adapter_path

    exports = ExportManager(
        python_exe=sys.executable, exports_dir=exports_dir(),
        run_resolver=resolve_run, push_fn=push_folder,
    )
    return Services(
        auth=auth,
        hub=HubClient(token_provider=auth.effective_token),
        downloads=DownloadManager(),
        cache=CacheService(),
        datasets=datasets,
        training=training,
        inference=InferenceService(),
        exports=exports,
    )
