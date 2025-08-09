from __future__ import annotations

from typing import Optional

import torch

_compiled = None


def _try_build() -> Optional[object]:
    global _compiled
    if _compiled is not None:
        return _compiled
    if not torch.cuda.is_available():
        return None
    try:
        import ute_qdq  # built via setup.py in ute/csrc
        _compiled = ute_qdq
        return _compiled
    except Exception:
        _compiled = None
        return None


def have_cuda_kernels() -> bool:
    return _try_build() is not None


def cuda_rowwise_quant_dequant(x: torch.Tensor, num_bits: int) -> torch.Tensor:
    mod = _try_build()
    if mod is None:
        raise RuntimeError("CUDA kernels not available")
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    rows, cols = x.shape[-2], x.shape[-1]
    x32 = x.contiguous().to(torch.float32)
    qmax = (1 << (num_bits - 1)) - 1
    scales = x32.abs().amax(dim=-1).clamp(min=1e-8) / float(qmax)
    y = torch.empty_like(x32)
    # Call compiled extension which computes scales internally on GPU
    y2 = mod.qdq_rowwise(x32.view(rows, cols), int(num_bits))
    if squeeze:
        return y2.squeeze(0)
    return y2


