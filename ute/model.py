from __future__ import annotations

import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, L, D)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class PreNormTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
        activation: Literal["gelu", "relu"] = "gelu",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False, attn_mask=attn_mask)
        x = x + self.dropout1(y)
        y = self.norm2(x)
        y = self.ff(y)
        x = x + self.dropout2(y)
        return x


class TransformerTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
        activation: Literal["gelu", "relu"] = "gelu",
        pooling: Literal["eos_token", "cls"] = "eos_token",
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len=max_sequence_length)
        self.layers = nn.ModuleList(
            [
                PreNormTransformerEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(embed_dim)
        self.pooling = pooling

    def forward(
        self, input_ids: torch.Tensor, eos_id: int, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input_ids: (B, L)
        x = self.embed_tokens(input_ids)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_out(x)

        # Compute pooled embedding
        if self.pooling == "eos_token":
            # Use last EOS position per sequence, fallback to last non-pad token
            with torch.no_grad():
                eos_mask = (input_ids == eos_id)
                # positions: last True index; if none, use last index where id != pad inferred by attention_mask
                indices = []
                for b in range(input_ids.size(0)):
                    pos = torch.nonzero(eos_mask[b], as_tuple=False)
                    if pos.numel() > 0:
                        indices.append(pos[-1].item())
                    else:
                        indices.append(int((attention_mask[b].sum() - 1).item()) if attention_mask is not None else input_ids.size(1) - 1)
                gather_idx = torch.tensor(indices, device=x.device)
            pooled = x[torch.arange(x.size(0), device=x.device), gather_idx]
        else:  # cls
            pooled = x[:, 0]
        return x, pooled


