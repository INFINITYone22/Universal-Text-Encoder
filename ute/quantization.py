from __future__ import annotations

from typing import Tuple

import torch

try:
    from .cuda_fp_kernels import have_cuda_kernels, cuda_rowwise_quant_dequant
except Exception:  # pragma: no cover
    def have_cuda_kernels() -> bool:
        return False

    def cuda_rowwise_quant_dequant(x: torch.Tensor, num_bits: int) -> torch.Tensor:  # type: ignore
        raise RuntimeError("CUDA kernels not available")


def _scale_and_clip(x: torch.Tensor, num_bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    qmax = 2 ** (num_bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    y = (x / scale).round().clamp(min=-qmax - 1, max=qmax)
    return y, scale


def simulate_fp8(x: torch.Tensor) -> torch.Tensor:
    """
    Simulate FP8 by per-vector quantize/dequantize using 8-bit signed integers.
    Not a true FP8 format; sufficient for research ablations.
    """
    q, s = _scale_and_clip(x, num_bits=8)
    return q * s


def simulate_fp4(x: torch.Tensor) -> torch.Tensor:
    q, s = _scale_and_clip(x, num_bits=4)
    return q * s


def maybe_quantize(x: torch.Tensor, precision: str) -> torch.Tensor:
    p = precision.lower()
    if p == "fp8":
        if x.is_cuda and have_cuda_kernels():
            return cuda_rowwise_quant_dequant(x, num_bits=8)
        return simulate_fp8(x)
    if p == "fp4":
        if x.is_cuda and have_cuda_kernels():
            return cuda_rowwise_quant_dequant(x, num_bits=4)
        return simulate_fp4(x)
    return x


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a * b).sum(dim=-1)


