import torch

from ute.quantization import simulate_fp4, simulate_fp8


def test_simulate_fp8_fp4_reduce_error():
    x = torch.randn(8, 16)
    y8 = simulate_fp8(x)
    y4 = simulate_fp4(x)
    err8 = (x - y8).abs().mean().item()
    err4 = (x - y4).abs().mean().item()
    assert err8 <= err4 + 1e-6


