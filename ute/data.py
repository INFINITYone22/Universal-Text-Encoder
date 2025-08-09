from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import CharacterTokenizer


@dataclass
class TextExample:
    text: str


class LineTextDataset(Dataset[TextExample]):
    def __init__(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing dataset file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> TextExample:
        return TextExample(text=self.lines[idx])


def mask_tokens(
    input_ids: torch.Tensor,
    mask_token_id: int,
    special_ids: Sequence[int],
    vocab_size: int,
    mlm_probability: float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability, device=input_ids.device)
    for sid in special_ids:
        probability_matrix = probability_matrix * (input_ids != sid)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    # 80% replace with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=input_ids.device)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id

    # 10% replace with random token
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5, device=input_ids.device)).bool() & masked_indices & ~indices_replaced
    )
    random_words = torch.randint(low=0, high=vocab_size, size=labels.shape, device=input_ids.device)
    input_ids[indices_random] = random_words[indices_random]

    # 10% keep original
    return input_ids, labels


class MLMDataCollator:
    def __init__(
        self,
        tokenizer: CharacterTokenizer,
        max_length: int,
        mlm_probability: float = 0.15,
    ) -> None:
        self.tok = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability

    def __call__(self, batch: List[TextExample]) -> Dict[str, torch.Tensor]:
        encoded = [
            self.tok.encode(ex.text, add_special_tokens=True, max_length=self.max_length)
            for ex in batch
        ]
        max_len = max(len(x) for x in encoded)
        input_ids = torch.full((len(encoded), max_len), self.tok.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
        for i, seq in enumerate(encoded):
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, : len(seq)] = 1

        masked_input_ids, labels = mask_tokens(
            input_ids=input_ids,
            mask_token_id=self.tok.mask_id,
            special_ids=[self.tok.pad_id, self.tok.bos_id, self.tok.eos_id],
            vocab_size=self.tok.vocab_size,
            mlm_probability=self.mlm_probability,
        )

        return {
            "input_ids": masked_input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class ContrastiveTextDataset(Dataset[Tuple[str, torch.Tensor]]):
    """
    Synthetic contrastive dataset: generates a deterministic pseudo-visual embedding
    for each text using a stable hash. This allows validating the training loop
    without external visual data.
    """

    def __init__(self, file_path: str, visual_dim: int = 128) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing dataset file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f.readlines() if line.strip()]
        self.visual_dim = int(visual_dim)

    def __len__(self) -> int:
        return len(self.lines)

    @staticmethod
    def _hashed_vector(text: str, dim: int) -> torch.Tensor:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Expand deterministically to required dim
        buf = bytearray()
        while len(buf) < dim * 4:
            h = hashlib.sha256(h).digest()
            buf.extend(h)
        arr = torch.frombuffer(bytes(buf[: dim * 4]), dtype=torch.uint8).float()
        arr = arr.view(-1, 4).sum(dim=1)
        vec = arr[:dim]
        vec = (vec - vec.mean()) / (vec.std() + 1e-6)
        return vec

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        text = self.lines[idx]
        vec = self._hashed_vector(text, self.visual_dim)
        return text, vec


class ContrastiveCollator:
    def __init__(self, tokenizer: CharacterTokenizer, max_length: int) -> None:
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Tuple[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        texts, visuals = zip(*batch)
        encoded = [self.tok.encode(t, add_special_tokens=True, max_length=self.max_length) for t in texts]
        max_len = max(len(x) for x in encoded)
        input_ids = torch.full((len(encoded), max_len), self.tok.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
        for i, seq in enumerate(encoded):
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, : len(seq)] = 1
        visuals_tensor = torch.stack(visuals, dim=0).float()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "visuals": visuals_tensor}


