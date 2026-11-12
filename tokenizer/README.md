# tokenizer

Two tokenizers written from scratch with only the Python standard library.

- `CharTokenizer` — one token per character, `<unk>` (id 0) for characters not seen in training. A baseline.
- `BPETokenizer` — byte-level byte-pair encoding. Starts from the 256 UTF-8 bytes, so any text round-trips. Text is pre-split GPT-2 style (words, numbers, punctuation, whitespace) and merges never cross those boundaries. Supports special tokens such as `<|endoftext|>`.

```python
from tokenizer import BPETokenizer

tok = BPETokenizer(special_tokens=["<|endoftext|>"])
tok.train(open("corpus.txt").read(), vocab_size=512)
ids = tok.encode("hello world<|endoftext|>")
assert tok.decode(ids) == "hello world<|endoftext|>"
tok.save("bpe.json")
tok = BPETokenizer.load("bpe.json")
```

Run the tests from the repo root with `python -m pytest tokenizer`.
