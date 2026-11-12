"""Common interface shared by every tokenizer in this package."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Tokenizer(ABC):
    """A tokenizer maps text to a list of integer ids and back."""

    @abstractmethod
    def train(self, text: str, vocab_size: int | None = None) -> None:
        """Learn a vocabulary from ``text``."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Turn ``text`` into token ids."""

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Turn token ids back into text."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Number of distinct token ids this tokenizer can produce."""

    # --- persistence -----------------------------------------------------

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialisable state, enough to rebuild the tokenizer."""

    @classmethod
    @abstractmethod
    def from_dict(cls, state: dict[str, Any]) -> "Tokenizer":
        """Rebuild a tokenizer from :meth:`to_dict` output."""

    def save(self, path: str | Path) -> None:
        state = {"type": type(self).__name__, **self.to_dict()}
        Path(path).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        kind = state.pop("type", cls.__name__)
        if kind != cls.__name__:
            raise ValueError(f"{path} holds a {kind}, not a {cls.__name__}")
        return cls.from_dict(state)
