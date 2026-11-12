import pytest

from tokenizer import CharTokenizer

TEXT = "hello world! héllo wörld 🙂\nnew line\ttab"


def test_round_trip():
    tok = CharTokenizer()
    tok.train(TEXT)
    assert tok.decode(tok.encode(TEXT)) == TEXT


def test_vocab_is_unk_plus_distinct_chars():
    tok = CharTokenizer()
    tok.train("abcabc")
    assert tok.vocab_size == 4
    assert tok.encode("cab") == [3, 1, 2]


def test_unknown_char_maps_to_unk():
    tok = CharTokenizer()
    tok.train("abc")
    assert tok.encode("axb") == [1, 0, 2]
    assert tok.decode(tok.encode("axb")) == "a<unk>b"


def test_vocab_size_keeps_most_frequent():
    tok = CharTokenizer()
    tok.train("aaaabbbc", vocab_size=3)
    assert tok.vocab_size == 3
    assert tok.encode("abc") == [1, 2, 0]


def test_vocab_size_must_leave_room_for_unk():
    with pytest.raises(ValueError):
        CharTokenizer().train("abc", vocab_size=0)


def test_save_load_round_trip(tmp_path):
    tok = CharTokenizer()
    tok.train(TEXT)
    path = tmp_path / "char.json"
    tok.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded.encode(TEXT) == tok.encode(TEXT)
    assert loaded.decode(loaded.encode(TEXT)) == TEXT
