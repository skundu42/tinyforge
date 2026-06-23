from tinyforge.train.models import LM_PRESETS, RunConfig, StartRunRequest, apply_preset


def test_engine_literal_allows_mlx_and_lm() -> None:
    assert RunConfig(name="r", model_repo="m", data_dir="/d", adapter_path="/a", engine="lm").engine == "lm"
    assert StartRunRequest(name="r", engine="lm").engine == "lm"


def test_lm_request_has_model_knobs_with_defaults() -> None:
    req = StartRunRequest(name="r", engine="lm")
    assert req.model_size == "small"
    assert req.hidden_size == 256
    assert req.num_heads == 8
    assert req.vocab_size == 8000
    assert req.context_length == 512


def test_apply_preset_expands_named_sizes() -> None:
    # tiny preset overrides whatever was passed
    assert apply_preset("tiny", 16, 999, 999, 999) == (4, 128, 4, 256)
    assert apply_preset("small", 16, 999, 999, 999) == (6, 256, 8, 512)
    assert apply_preset("medium", 16, 999, 999, 999) == (8, 512, 8, 512)


def test_apply_preset_custom_passes_values_through() -> None:
    assert apply_preset("custom", 5, 384, 6, 1024) == (5, 384, 6, 1024)
