from tinyforge.train.tokenizer import train_bpe


def test_train_bpe_roundtrips_text() -> None:
    corpus = ["hello world", "hello there", "world of words"] * 20
    tok = train_bpe(corpus, vocab_size=300)
    ids = tok("hello world")["input_ids"]
    assert len(ids) > 0
    assert "hello" in tok.decode(ids)


def test_train_bpe_sets_special_tokens() -> None:
    tok = train_bpe(["abc def ghi"] * 20, vocab_size=300)
    assert tok.eos_token == "<|endoftext|>"
    assert tok.pad_token == "<|endoftext|>"
    assert tok.eos_token_id is not None
