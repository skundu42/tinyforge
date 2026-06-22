"""Tests for HubClient search/detail mapping (HfApi behind a fake)."""

from datetime import datetime
from types import SimpleNamespace

from tinyforge.hub.client import HubClient


class FakeApi:
    def __init__(self, models=None, datasets=None, info=None):
        self._models = models or []
        self._datasets = datasets or []
        self._info = info
        self.calls: dict[str, dict] = {}

    def list_models(self, **kwargs):
        self.calls["list_models"] = kwargs
        return iter(self._models)

    def list_datasets(self, **kwargs):
        self.calls["list_datasets"] = kwargs
        return iter(self._datasets)

    def model_info(self, repo_id, **kwargs):
        self.calls["model_info"] = {"repo_id": repo_id, **kwargs}
        return self._info


def make_model(model_id="meta/x", **kw):
    base = dict(
        id=model_id, author="meta", downloads=100, likes=5, gated=False,
        private=False, pipeline_tag="text-generation", library_name="transformers",
        tags=["a"], last_modified=datetime(2026, 1, 2),
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_search_models_maps_fields_and_passes_query() -> None:
    api = FakeApi(models=[make_model(model_id="mlx-community/Llama-3.2-1B-4bit", gated="manual")])
    client = HubClient(api=api, token_provider=lambda: None)

    results = client.search_models(query="llama", sort="downloads", limit=10)

    assert api.calls["list_models"]["search"] == "llama"
    assert api.calls["list_models"]["sort"] == "downloads"
    assert api.calls["list_models"]["limit"] == 10
    assert len(results) == 1
    model = results[0]
    assert model.id == "mlx-community/Llama-3.2-1B-4bit"
    assert model.gated is True  # normalized from "manual"
    assert model.pipeline_tag == "text-generation"
    assert model.last_modified is not None and model.last_modified.startswith("2026-01-02")


def test_search_models_passes_effective_token() -> None:
    api = FakeApi(models=[])
    client = HubClient(api=api, token_provider=lambda: "tok")
    client.search_models(query=None)
    assert api.calls["list_models"]["token"] == "tok"


def test_search_datasets_maps_fields() -> None:
    dataset = SimpleNamespace(
        id="squad", author="rajpurkar", downloads=50, likes=3, gated=False,
        private=False, tags=["qa"], last_modified=datetime(2026, 1, 1),
    )
    api = FakeApi(datasets=[dataset])
    client = HubClient(api=api, token_provider=lambda: None)

    results = client.search_datasets(query="squad")

    assert results[0].id == "squad"
    assert results[0].downloads == 50


def test_model_detail_includes_siblings_total_and_readme() -> None:
    info = make_model(model_id="meta/x")
    info.siblings = [
        SimpleNamespace(rfilename="model.safetensors", size=1234),
        SimpleNamespace(rfilename="config.json", size=10),
    ]
    api = FakeApi(info=info)
    client = HubClient(
        api=api, token_provider=lambda: None,
        readme_fn=lambda repo_id, token=None: "# Hello",
    )

    detail = client.model_detail("meta/x")

    assert detail.id == "meta/x"
    assert {f.filename for f in detail.siblings} == {"model.safetensors", "config.json"}
    assert detail.total_size == 1244
    assert detail.readme == "# Hello"


def test_model_detail_survives_readme_failure() -> None:
    info = make_model(model_id="meta/x")
    info.siblings = []

    def boom(repo_id, token=None):
        raise OSError("no readme")

    client = HubClient(api=FakeApi(info=info), token_provider=lambda: None, readme_fn=boom)
    detail = client.model_detail("meta/x")
    assert detail.readme is None
