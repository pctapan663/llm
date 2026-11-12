import pytest

from tokenizer import BPETokenizer, CharTokenizer

CORPUS = (
    "The quick brown fox jumps over the lazy dog. "
    "The dog sleeps; the fox runs. It's 2026 and the year 12345 is far off.\n"
) * 20

SAMPLES = [
    "",
    "a",
    "hello world",
    "The quick brown fox jumps over the lazy dog.",
    "  leading and trailing spaces   ",
    "tabs\tand\nnewlines\r\n",
    "unicode: héllo wörld, 日本語, emoji 🙂👍🏽",
    "numbers 1234567890 and punctuation!?..;:'\"",
    "text never seen in training: zyxwvut qqq",
]


@pytest.fixture(scope="module")
def trained():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=306)
    return tok


@pytest.mark.parametrize("text", SAMPLES)
def test_round_trip(trained, text):
    assert trained.decode(trained.encode(text)) == text


def test_untrained_is_plain_utf8_bytes():
    tok = BPETokenizer()
    text = "héllo 🙂"
    assert tok.encode(text) == list(text.encode("utf-8"))
    assert tok.decode(tok.encode(text)) == text


def test_learns_requested_number_of_merges(trained):
    assert len(trained.merges) == 50
    assert trained.vocab_size == 306


def test_compresses_training_text(trained):
    assert len(trained.encode(CORPUS)) < len(CORPUS.encode("utf-8")) / 2


def test_first_merge_is_most_frequent_pair():
    tok = BPETokenizer()
    tok.train("aaabdaaabac", vocab_size=257)
    # "aaabdaaabac" is one chunk; ("a","a") occurs 4 times, more than any pair.
    assert tok.merges == {(ord("a"), ord("a")): 256}
    assert tok.vocab[256] == b"aa"


def test_merges_do_not_cross_chunk_boundaries(trained):
    for token in trained.vocab.values():
        text = token.decode("utf-8", errors="ignore")
        # No learned token joins a letter with following punctuation, e.g. "dog."
        assert not any(a.isalpha() and b in ".;" for a, b in zip(text, text[1:]))


def test_training_is_deterministic():
    a, b = BPETokenizer(), BPETokenizer()
    a.train(CORPUS, vocab_size=300)
    b.train(CORPUS, vocab_size=300)
    assert a.merges == b.merges


def test_stops_early_when_nothing_repeats():
    tok = BPETokenizer()
    tok.train("abcdef", vocab_size=1000)
    assert tok.merges == {}


def test_vocab_size_below_bytes_rejected():
    with pytest.raises(ValueError):
        BPETokenizer().train(CORPUS, vocab_size=100)


def test_special_tokens(trained):
    tok = BPETokenizer(special_tokens=["<|endoftext|>"])
    tok.train(CORPUS, vocab_size=300)
    eot = tok.special_tokens["<|endoftext|>"]
    assert eot == 300
    text = "the fox<|endoftext|>the dog"
    ids = tok.encode(text)
    assert ids.count(eot) == 1
    assert tok.decode(ids) == text


def test_decode_rejects_unknown_id(trained):
    with pytest.raises(ValueError):
        trained.decode([99999])


def test_save_load_round_trip(tmp_path):
    tok = BPETokenizer(special_tokens=["<|endoftext|>"])
    tok.train(CORPUS, vocab_size=320)
    path = tmp_path / "bpe.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    for text in SAMPLES + ["x<|endoftext|>y"]:
        assert loaded.encode(text) == tok.encode(text)
        assert loaded.decode(loaded.encode(text)) == text


def test_load_rejects_wrong_type(tmp_path):
    tok = CharTokenizer()
    tok.train("abc")
    path = tmp_path / "char.json"
    tok.save(path)
    with pytest.raises(ValueError):
        BPETokenizer.load(path)
