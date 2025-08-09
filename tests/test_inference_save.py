import os
from pathlib import Path

import torch
from safetensors.torch import load_file

from ute.model import TransformerTextEncoder
from ute.tokenizer import CharacterTokenizer
from ute.inference import build_from_checkpoint
from ute.utils import save_checkpoint


def test_build_from_checkpoint_and_save(tmp_path: Path):
    # Minimal config and checkpoint
    cfg_text = (
        "model:\n"
        "  type: transformer_encoder\n"
        "  precision: bf16\n"
        "  vocab_size: 64\n"
        "  max_sequence_length: 64\n"
        "  layers: 1\n"
        "  attention_heads: 4\n"
        "  hidden_dim: 32\n"
        "  feedforward_dim: 64\n"
        "  activation: gelu\n"
        "  dropout: 0.0\n"
        "output:\n"
        "  pooling: eos_token\n"
        "  embedding_dim: 32\n"
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")

    tok = CharacterTokenizer(vocab_size=64)
    model = TransformerTextEncoder(
        vocab_size=64,
        max_sequence_length=64,
        embed_dim=32,
        num_layers=1,
        num_heads=4,
        feedforward_dim=64,
        dropout=0.0,
    )
    ckpt_dir = tmp_path / "run"
    ckpt_dir.mkdir()
    ckpt_path = save_checkpoint(
        output_dir=str(ckpt_dir),
        model_state={"encoder": model.state_dict(), "tokenizer": tok.to_serializable()},
        filename="last.pt",
    )

    tok2, model2 = build_from_checkpoint(str(cfg_path), ckpt_path)
    assert isinstance(tok2, CharacterTokenizer)
    assert isinstance(model2, TransformerTextEncoder)

    # Prepare inputs
    texts = ["hello", "world!"]
    ids = [tok2.encode(t) for t in texts]
    max_len = max(len(x) for x in ids)
    input_ids = torch.full((len(texts), max_len), tok2.pad_id, dtype=torch.long)
    attn = torch.zeros_like(input_ids)
    for i, seq in enumerate(ids):
        input_ids[i, : len(seq)] = torch.tensor(seq)
        attn[i, : len(seq)] = 1

    with torch.no_grad():
        _, pooled = model2(input_ids=input_ids, eos_id=tok2.eos_id, attention_mask=attn)
    out_path = tmp_path / "emb.safetensors"
    from safetensors.torch import save_file

    import json
    save_file({"embeddings": pooled}, str(out_path), metadata={"texts": json.dumps(texts)})
    loaded = load_file(str(out_path))
    assert "embeddings" in loaded
    emb = loaded["embeddings"]
    assert list(emb.shape)[0] == len(texts)


