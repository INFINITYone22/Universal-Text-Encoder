import torch

from ute.model import TransformerTextEncoder
from ute.tokenizer import CharacterTokenizer


def test_model_forward_shapes():
    tok = CharacterTokenizer(vocab_size=128)
    model = TransformerTextEncoder(
        vocab_size=128,
        max_sequence_length=64,
        embed_dim=32,
        num_layers=1,
        num_heads=4,
        feedforward_dim=64,
        dropout=0.0,
    )
    texts = ["abc", "abcd"]
    encoded = [tok.encode(t) for t in texts]
    max_len = max(len(x) for x in encoded)
    input_ids = torch.full((len(texts), max_len), tok.pad_id, dtype=torch.long)
    attn = torch.zeros_like(input_ids)
    for i, seq in enumerate(encoded):
        input_ids[i, : len(seq)] = torch.tensor(seq)
        attn[i, : len(seq)] = 1
    hidden, pooled = model(input_ids=input_ids, eos_id=tok.eos_id, attention_mask=attn)
    assert hidden.shape[:2] == input_ids.shape
    assert pooled.shape[0] == input_ids.shape[0]


