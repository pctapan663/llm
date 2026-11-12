"""Character-level tokenizer: one token per Unicode code point.

The simplest possible tokenizer and a useful baseline. The vocabulary is just
the set of distinct characters seen during training. Characters that never
appeared in training map to a reserved ``<unk>`` id (0), so they cannot be
decoded back to the original text.
"""

from __future__ import annotations

from typing import Any

from tokenizer.base import Tokenizer

UNK = "<unk>"


class CharTokenizer(Tokenizer):
    def __init__(self) -> None:
        self.itos: list[str] = [UNK]
        self.stoi: dict[str, int] = {UNK: 0}

    def train(self, text: str, vocab_size: int | None = None) -> None:
        """Build the vocabulary from the distinct characters in ``text``.

        If ``vocab_size`` is given, keep only the ``vocab_size - 1`` most
        frequent characters (one slot is reserved for ``<unk>``).
        """
        counts: dict[str, int] = {}
        for ch in text:
            counts[ch] = counts.get(ch, 0) + 1
        chars = sorted(counts)  # deterministic order
        if vocab_size is not None:
            if vocab_size < 1:
                raise ValueError("vocab_size must be at least 1 (for <unk>)")
            keep = sorted(counts, key=lambda c: (-counts[c], c))[: vocab_size - 1]
            chars = sorted(keep)
        self.itos = [UNK, *chars]
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def to_dict(self) -> dict[str, Any]:
        return {"chars": self.itos[1:]}

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "CharTokenizer":
        tok = cls()
        tok.itos = [UNK, *state["chars"]]
        tok.stoi = {ch: i for i, ch in enumerate(tok.itos)}
        return tok
