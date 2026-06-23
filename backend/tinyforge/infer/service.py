"""Streaming text generation via mlx_lm, with optional LoRA adapter.

The model is loaded on demand and cached (one at a time, evicted when a
different model/adapter is requested) so repeated prompts are fast. The
generator is injectable so the WebSocket plumbing is testable without mlx.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator

from pydantic import BaseModel


class GenRequest(BaseModel):
    model_repo: str
    adapter_path: str | None = None
    prompt: str
    max_tokens: int = 256
    temp: float = 0.7
    top_p: float = 0.9
    chat: bool = True


# Single-entry model cache: (repo, adapter) -> (model, tokenizer)
_CACHE: dict[tuple[str, str | None], tuple] = {}


def _load_cached(repo: str, adapter_path: str | None):
    import mlx.core as mx
    from mlx_lm import load

    key = (repo, adapter_path)
    if key not in _CACHE:
        _CACHE.clear()  # keep only one model resident
        mx.clear_cache()
        _CACHE[key] = load(repo, adapter_path=adapter_path)
    return _CACHE[key]


def _build_prompt(tokenizer, prompt: str, chat: bool) -> str:
    if chat and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False,
        )
    return prompt


def _default_generate(req: GenRequest) -> Iterator[str]:
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = _load_cached(req.model_repo, req.adapter_path)
    prompt = _build_prompt(tokenizer, req.prompt, req.chat)
    sampler = make_sampler(temp=req.temp, top_p=req.top_p)
    for response in stream_generate(
        model, tokenizer, prompt, max_tokens=req.max_tokens, sampler=sampler
    ):
        if response.text:
            yield response.text
        if response.finish_reason:
            break


class InferenceService:
    def __init__(self, generate_fn: Callable[[GenRequest], Iterator[str]] = _default_generate) -> None:
        self._generate = generate_fn
        # The single-entry model cache and mlx itself are not thread-safe, so
        # generations are serialized (e.g. base-vs-finetuned compare runs one at
        # a time rather than corrupting the cache mid-eval).
        self._lock = threading.Lock()

    def stream(
        self, request: GenRequest, should_cancel: Callable[[], bool] = lambda: False
    ) -> Iterator[str]:
        """Yield generated text deltas, stopping early when `should_cancel()` is true.

        Cancellation stops pulling from the underlying generator, so the GPU work
        suspends between tokens; the generator is then closed so its own cleanup
        (releasing the lock / mlx resources) runs promptly.
        """
        with self._lock:
            generated = self._generate(request)
            try:
                for delta in generated:
                    if should_cancel():
                        break
                    yield delta
            finally:
                close = getattr(generated, "close", None)
                if callable(close):
                    close()
