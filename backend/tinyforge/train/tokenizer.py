"""Train a small byte-level BPE tokenizer on the user's corpus (truly from scratch).

Produces a `PreTrainedTokenizerFast` that `save_pretrained`s into a standard
`tokenizer.json` + `tokenizer_config.json`, so the trained model round-trips in
`transformers` and `mlx_lm`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

EOS = "<|endoftext|>"


def train_bpe(texts: Iterable[str], vocab_size: int) -> "PreTrainedTokenizerFast":
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(list(texts), trainer)

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token=EOS,
        pad_token=EOS,
        bos_token=EOS,
        unk_token=EOS,
    )
