from ute.tokenizer import CharacterTokenizer


def test_tokenizer_encode_decode():
    tok = CharacterTokenizer(vocab_size=64)
    text = "Hello   world!"
    ids = tok.encode(text, add_special_tokens=True, max_length=128)
    out = tok.decode(ids, skip_special_tokens=True)
    assert "  " not in out
    assert out.replace(" ", "") == "Helloworld!"


