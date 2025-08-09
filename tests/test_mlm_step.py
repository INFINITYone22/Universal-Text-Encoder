import torch

from ute.train_mlm import MLMHead
from ute.model import TransformerTextEncoder
from ute.tokenizer import CharacterTokenizer


def test_mlm_forward_loss():
    vocab = 64
    tok = CharacterTokenizer(vocab_size=vocab)
    model = TransformerTextEncoder(
        vocab_size=vocab,
        max_sequence_length=32,
        embed_dim=16,
        num_layers=1,
        num_heads=4,
        feedforward_dim=32,
        dropout=0.0,
    )
    head = MLMHead(hidden_dim=16, vocab_size=vocab)

    texts = ["hello", "world"]
    encoded = [tok.encode(t) for t in texts]
    max_len = max(len(x) for x in encoded)
    input_ids = torch.full((len(texts), max_len), tok.pad_id, dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    attn = torch.zeros_like(input_ids)
    for i, seq in enumerate(encoded):
        input_ids[i, : len(seq)] = torch.tensor(seq)
        attn[i, : len(seq)] = 1
        # mask one middle token if possible
        if len(seq) > 3:
            labels[i, 2] = input_ids[i, 2]
            input_ids[i, 2] = tok.mask_id

    hidden, _ = model(input_ids=input_ids, eos_id=tok.eos_id, attention_mask=attn)
    logits = head(hidden)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
    assert torch.isfinite(loss)


