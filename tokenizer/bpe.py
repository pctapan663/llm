"""Byte-level byte-pair encoding (BPE) tokenizer, written from scratch.

How it works
------------
1. Text is UTF-8 encoded, so the base vocabulary is the 256 possible bytes.
   Any string can be encoded, and nothing is ever "unknown".
2. Text is first split into chunks (words, numbers, punctuation runs,
   whitespace) with a regex, in the style of GPT-2. Merges never cross chunk
   boundaries, which stops tokens like "dog." or "the cat" from forming.
3. Training repeatedly finds the most frequent adjacent pair of token ids and
   replaces it with a new id, until the vocabulary reaches ``vocab_size``.
4. Encoding replays the learned merges in the order they were learned.
5. Decoding concatenates each token's bytes and decodes them as UTF-8.

Special tokens (e.g. ``<|endoftext|>``) can be registered; they are matched
literally during encoding and are never split or merged.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from tokenizer.base import Tokenizer

# GPT-2 style pre-tokenisation using only the stdlib ``re`` module.
# [^\W\d_] means "a letter" (word char that is not a digit or underscore).
SPLIT_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)"   # English contractions
    r"| ?[^\W\d_]+"           # words, with an optional leading space
    r"| ?\d{1,3}"             # numbers, at most 3 digits per chunk
    r"| ?[^\s\w]+"            # punctuation / symbols
    r"|\s+(?!\S)"             # trailing whitespace
    r"|\s+"                   # any other whitespace
)

Pair = tuple[int, int]


def _pair_counts(chunks: dict[tuple[int, ...], int]) -> Counter[Pair]:
    counts: Counter[Pair] = Counter()
    for ids, freq in chunks.items():
        for pair in zip(ids, ids[1:]):
            counts[pair] += freq
    return counts


def _merge(ids: tuple[int, ...] | list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` in ``ids``."""
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i + 1 < n and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer(Tokenizer):
    def __init__(self, pattern: str = SPLIT_PATTERN, special_tokens: list[str] | None = None) -> None:
        self.pattern = pattern
        self._regex = re.compile(pattern)
        self.merges: dict[Pair, int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}
        for tok in special_tokens or []:
            self.add_special_token(tok)

    # --- training --------------------------------------------------------

    def train(self, text: str, vocab_size: int | None = None, verbose: bool = False) -> None:
        """Learn ``vocab_size - 256`` merges from ``text``.

        Training resets any merges learned before. Special tokens are kept and
        get ids after the merged tokens. Training stops early if no pair occurs
        at least twice, since merging a pair seen once does not compress.
        """
        if vocab_size is None:
            raise ValueError("BPETokenizer.train needs a vocab_size")
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (the byte alphabet)")

        specials = list(self.special_tokens)
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_tokens = {}

        # Count each distinct chunk once, so repeated words cost nothing extra.
        chunk_freq = Counter(self._regex.findall(text))
        chunks: dict[tuple[int, ...], int] = {
            tuple(c.encode("utf-8")): f for c, f in chunk_freq.items()
        }

        for new_id in range(256, vocab_size):
            counts = _pair_counts(chunks)
            if not counts:
                break
            # Most frequent pair; ties broken by the smallest pair for determinism.
            pair, freq = max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if freq < 2:
                break
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            merged: dict[tuple[int, ...], int] = {}
            for ids, f in chunks.items():
                key = tuple(_merge(ids, pair, new_id)) if len(ids) > 1 else ids
                merged[key] = merged.get(key, 0) + f
            chunks = merged
            if verbose:
                print(f"merge {new_id - 255}: {pair} -> {new_id} {self.vocab[new_id]!r} ({freq}x)")

        for tok in specials:
            self.add_special_token(tok)

    def add_special_token(self, token: str) -> int:
        if token in self.special_tokens:
            return self.special_tokens[token]
        # Ids are contiguous: bytes, then merges, then specials.
        new_id = len(self.vocab) + len(self.special_tokens)
        self.special_tokens[token] = new_id
        return new_id

    # --- encoding / decoding --------------------------------------------

    def _encode_chunk(self, chunk: bytes) -> list[int]:
        ids = list(chunk)
        while len(ids) >= 2:
            # Apply the earliest-learned merge present in this chunk.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in self._regex.findall(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self._encode_ordinary(text)
        # Longest specials first so overlapping ones match greedily.
        alts = sorted(self.special_tokens, key=len, reverse=True)
        special_re = "(" + "|".join(re.escape(s) for s in alts) + ")"
        ids: list[int] = []
        for part in re.split(special_re, text):
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self._encode_ordinary(part))
        return ids

    def decode(self, ids: list[int]) -> str:
        inverse_special = {i: s.encode("utf-8") for s, i in self.special_tokens.items()}
        parts: list[bytes] = []
        for i in ids:
            if i in self.vocab:
                parts.append(self.vocab[i])
            elif i in inverse_special:
                parts.append(inverse_special[i])
            else:
                raise ValueError(f"unknown token id {i}")
        # A slice of ids may cut a multi-byte character; don't crash on it.
        return b"".join(parts).decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    # --- persistence -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        # Merges are stored in learned order; ids follow from that order.
        ordered = sorted(self.merges.items(), key=lambda kv: kv[1])
        return {
            "pattern": self.pattern,
            "merges": [[a, b] for (a, b), _ in ordered],
            "special_tokens": self.special_tokens,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "BPETokenizer":
        tok = cls(pattern=state["pattern"])
        for idx, (a, b) in enumerate(state["merges"]):
            new_id = 256 + idx
            tok.merges[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        tok.special_tokens = {s: int(i) for s, i in state["special_tokens"].items()}
        return tok
