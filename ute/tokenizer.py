from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


class CharacterTokenizer:
    """
    Character-level tokenizer with configurable vocabulary size and special tokens.
    If a vocabulary mapping is not provided, the tokenizer builds a dynamic
    mapping for seen characters, reserving space up to vocab_size.
    """

    PAD = "[PAD]"
    BOS = "[BOS]"
    EOS = "[EOS]"
    MASK = "[MASK]"
    UNK = "[UNK]"

    def __init__(
        self,
        vocab_size: int,
        space_normalization: bool = True,
        initial_vocab: Optional[Dict[str, int]] = None,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.space_normalization = space_normalization

        if initial_vocab is None:
            self.char_to_id: Dict[str, int] = {}
            self._init_special_tokens()
        else:
            self.char_to_id = dict(initial_vocab)

        self.id_to_char: Dict[int, str] = {i: s for s, i in self.char_to_id.items()}

    def _init_special_tokens(self) -> None:
        # Reserve leading ids for special tokens
        specials = [self.PAD, self.BOS, self.EOS, self.MASK, self.UNK]
        self.char_to_id = {s: i for i, s in enumerate(specials)}

    @property
    def pad_id(self) -> int:
        return self.char_to_id[self.PAD]

    @property
    def bos_id(self) -> int:
        return self.char_to_id[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.char_to_id[self.EOS]

    @property
    def mask_id(self) -> int:
        return self.char_to_id[self.MASK]

    @property
    def unk_id(self) -> int:
        return self.char_to_id[self.UNK]

    def normalize(self, text: str) -> str:
        if self.space_normalization:
            text = re.sub(r"\s+", " ", text)
        return text

    def _assign_id(self, ch: str) -> int:
        if ch in self.char_to_id:
            return self.char_to_id[ch]
        next_id = len(self.char_to_id)
        if next_id >= self.vocab_size:
            return self.unk_id
        self.char_to_id[ch] = next_id
        self.id_to_char[next_id] = ch
        return next_id

    def encode(
        self, text: str, add_special_tokens: bool = True, max_length: Optional[int] = None
    ) -> List[int]:
        text = self.normalize(text)
        token_ids: List[int] = []
        if add_special_tokens:
            token_ids.append(self.bos_id)
        for ch in text:
            token_ids.append(self._assign_id(ch))
        if add_special_tokens:
            token_ids.append(self.eos_id)
        if max_length is not None:
            token_ids = token_ids[:max_length]
        return token_ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        chars: List[str] = []
        for i in ids:
            ch = self.id_to_char.get(i, "")
            if skip_special_tokens and ch in {
                self.PAD,
                self.BOS,
                self.EOS,
                self.MASK,
                self.UNK,
            }:
                continue
            chars.append(ch)
        return "".join(chars)

    def to_serializable(self) -> Dict[str, int]:
        return dict(self.char_to_id)

    @classmethod
    def from_serializable(
        cls, vocab_size: int, mapping: Dict[str, int], space_normalization: bool = True
    ) -> "CharacterTokenizer":
        return cls(
            vocab_size=vocab_size,
            space_normalization=space_normalization,
            initial_vocab=mapping,
        )


