import json

from tinyforge.train.lm_data import load_corpus, pack_tokens, render_text


def test_render_text_handles_each_format() -> None:
    assert render_text({"text": "hello"}) == "hello"
    assert render_text({"prompt": "Q?", "completion": "A."}) == "Q?\n\nA."
    assert render_text(
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    ) == "user: hi\nassistant: yo"


def test_pack_tokens_concatenates_with_eos_and_chunks() -> None:
    # two docs [1,2] and [3]; eos=0; block=3 -> [1,2,0],[3,0] (last short chunk dropped)
    blocks = pack_tokens([[1, 2], [3]], block_size=3, eos_id=0)
    assert blocks == [[1, 2, 0]]


def test_pack_tokens_keeps_all_full_blocks() -> None:
    blocks = pack_tokens([[1, 1, 1], [2, 2]], block_size=2, eos_id=9)
    # stream: 1,1,1,9,2,2,9 -> blocks of 2: [1,1],[1,9],[2,2]; trailing [9] dropped
    assert blocks == [[1, 1], [1, 9], [2, 2]]


def test_load_corpus_reads_jsonl(tmp_path) -> None:
    (tmp_path / "train.jsonl").write_text(json.dumps({"text": "a"}) + "\n" + json.dumps({"text": "b"}) + "\n")
    (tmp_path / "valid.jsonl").write_text(json.dumps({"text": "c"}) + "\n")
    train, val = load_corpus(str(tmp_path))
    assert train == ["a", "b"]
    assert val == ["c"]
