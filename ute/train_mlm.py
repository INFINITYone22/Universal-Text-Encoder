from __future__ import annotations

import argparse
import math
import os
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import LineTextDataset, MLMDataCollator
from .model import TransformerTextEncoder
from .tokenizer import CharacterTokenizer
from .utils import Config, amp_dtype_from_precision, ensure_dir, get_torch_device, save_checkpoint, set_all_seeds


class MLMHead(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


def build_model_and_tokenizer(cfg: Config) -> Dict[str, object]:
    vocab_size = int(cfg.get("model.vocab_size"))
    max_len = int(cfg.get("model.max_sequence_length"))
    layers = int(cfg.get("model.layers"))
    heads = int(cfg.get("model.attention_heads"))
    hidden = int(cfg.get("model.hidden_dim"))
    ff = int(cfg.get("model.feedforward_dim"))
    dropout = float(cfg.get("model.dropout", 0.1))
    activation = cfg.get("model.activation", "gelu")
    pooling = cfg.get("output.pooling", "eos_token")
    tok = CharacterTokenizer(vocab_size=vocab_size, space_normalization=True)
    model = TransformerTextEncoder(
        vocab_size=vocab_size,
        max_sequence_length=max_len,
        embed_dim=hidden,
        num_layers=layers,
        num_heads=heads,
        feedforward_dim=ff,
        dropout=dropout,
        activation=activation,
        pooling=pooling,
    )
    head = MLMHead(hidden_dim=hidden, vocab_size=vocab_size)
    return {"tokenizer": tok, "encoder": model, "mlm_head": head}


def train(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    set_all_seeds(int(cfg.get("training.seed", 42)))
    device = get_torch_device()
    precision = str(cfg.get("model.precision", "bf16"))
    amp_dtype = amp_dtype_from_precision(precision)

    built = build_model_and_tokenizer(cfg)
    tok: CharacterTokenizer = built["tokenizer"]  # type: ignore
    encoder: TransformerTextEncoder = built["encoder"]  # type: ignore
    head: MLMHead = built["mlm_head"]  # type: ignore
    encoder.to(device)
    head.to(device)

    data_path = args.data or os.path.join(os.path.dirname(__file__), "..", "data", "toy.txt")
    dataset = LineTextDataset(os.path.abspath(data_path))
    collate = MLMDataCollator(tokenizer=tok, max_length=int(cfg.get("model.max_sequence_length")))
    loader = DataLoader(dataset, batch_size=int(cfg.get("training.batch_size", 32)), shuffle=True, collate_fn=collate)

    optim = AdamW(list(encoder.parameters()) + list(head.parameters()), lr=float(cfg.get("training.learning_rate", 5e-4)), weight_decay=float(cfg.get("training.weight_decay", 0.01)))
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    encoder.train()
    head.train()
    steps = 0
    max_steps = int(args.max_steps) if args.max_steps else None
    eval_interval = int(args.eval_interval) if args.eval_interval else 0

    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype is not None and device.type == "cuda")

    output_dir = args.output_dir or "runs/tiny_mlm"
    ensure_dir(output_dir)

    with tqdm(total=max_steps if max_steps else len(loader) * int(cfg.get("training.epochs", 1))) as pbar:
        for epoch in range(int(cfg.get("training.epochs", 1))):
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                optim.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    hidden, _ = encoder(input_ids=input_ids, eos_id=tok.eos_id, attention_mask=attention_mask)
                    logits = head(hidden)
                    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), float(cfg.get("training.gradient_clipping", 1.0)))
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), float(cfg.get("training.gradient_clipping", 1.0)))
                    optim.step()

                steps += 1
                pbar.set_description(f"loss={loss.item():.4f}")
                pbar.update(1)

                if eval_interval and steps % eval_interval == 0:
                    save_checkpoint(
                        output_dir,
                        model_state={"encoder": encoder.state_dict(), "head": head.state_dict(), "tokenizer": tok.to_serializable()},
                        optimizer_state=optim.state_dict(),
                        extra={"step": steps, "epoch": epoch},
                        filename="last.pt",
                    )

                if max_steps and steps >= max_steps:
                    break
            if max_steps and steps >= max_steps:
                break

    save_checkpoint(
        output_dir,
        model_state={"encoder": encoder.state_dict(), "head": head.state_dict(), "tokenizer": tok.to_serializable()},
        optimizer_state=optim.state_dict(),
        extra={"final": True, "steps": steps},
        filename="last.pt",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--data", type=str, default="")
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--eval_interval", type=int, default=0)
    ap.add_argument("--output_dir", type=str, default="")
    args = ap.parse_args()
    train(args)


