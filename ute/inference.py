from __future__ import annotations

import argparse
import os
from typing import List

import torch
from safetensors.torch import save_file
import json

from .model import TransformerTextEncoder
from .quantization import maybe_quantize
from .tokenizer import CharacterTokenizer
from .utils import Config, amp_dtype_from_precision, get_torch_device, load_checkpoint


def build_from_checkpoint(cfg_path: str, ckpt_path: str) -> tuple[CharacterTokenizer, TransformerTextEncoder]:
    cfg = Config.load(cfg_path)
    vocab_size = int(cfg.get("model.vocab_size"))
    max_len = int(cfg.get("model.max_sequence_length"))
    layers = int(cfg.get("model.layers"))
    heads = int(cfg.get("model.attention_heads"))
    hidden = int(cfg.get("model.hidden_dim"))
    ff = int(cfg.get("model.feedforward_dim"))
    dropout = float(cfg.get("model.dropout", 0.1))
    activation = cfg.get("model.activation", "gelu")
    pooling = cfg.get("output.pooling", "eos_token")

    bundle = load_checkpoint(ckpt_path)
    tok_map = bundle["model"].get("tokenizer") if "model" in bundle else bundle.get("tokenizer")
    tok = CharacterTokenizer.from_serializable(vocab_size=vocab_size, mapping=tok_map, space_normalization=True)

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
    state = bundle["model"].get("encoder") if "model" in bundle else bundle.get("encoder")
    model.load_state_dict(state)
    return tok, model


def run_inference(args: argparse.Namespace) -> None:
    device = get_torch_device()
    cfg = Config.load(args.config)
    precision = str(args.precision or cfg.get("model.precision", "bf16"))
    amp_dtype = amp_dtype_from_precision(precision)
    tok, model = build_from_checkpoint(args.config, args.checkpoint)
    model.to(device)
    model.eval()

    texts: List[str] = args.texts
    input_ids_list = [tok.encode(t, add_special_tokens=True, max_length=int(cfg.get("model.max_sequence_length"))) for t in texts]
    max_len = max(len(x) for x in input_ids_list)
    input_ids = torch.full((len(texts), max_len), tok.pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(texts), max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(input_ids_list):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask[i, : len(seq)] = 1

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        _, pooled = model(input_ids=input_ids, eos_id=tok.eos_id, attention_mask=attention_mask)
        pooled = maybe_quantize(pooled, precision)

    tensors = {"embeddings": pooled.detach().cpu()}
    metadata = {"texts": json.dumps(texts, ensure_ascii=False)}
    save_file(tensors, args.out, metadata=metadata)
    print(f"Saved embeddings to {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--texts", nargs="+", type=str, required=True)
    ap.add_argument("--precision", type=str, default="")
    run_inference(ap.parse_args())


