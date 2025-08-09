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
        from torch.utils.cpp_extension import load_inline

        cuda_src = r"""
        extern "C" __global__ void quant_dequant_rowwise(
            const float* __restrict__ x,
            const float* __restrict__ scales, // size: rows
            float* __restrict__ y,
            const long rows,
            const long cols,
            const int qmax
        ) {
            long row = blockIdx.x;
            if (row >= rows) return;
            float scale = scales[row];
            long start = row * cols;
            for (long i = threadIdx.x; i < cols; i += blockDim.x) {
                float v = x[start + i] / scale;
                // round to nearest
                v = nearbyintf(v);
                // clamp symmetrically; allow -qmax-1 for signed
                float lo = -float(qmax) - 1.0f;
                float hi = float(qmax);
                if (v < lo) v = lo;
                if (v > hi) v = hi;
                y[start + i] = v * scale;
            }
        }
        """

        _compiled = load_inline(
            name="ute_fp_kernels",
            cpp_sources="",
            cuda_sources=cuda_src,
            functions=["quant_dequant_rowwise"],
            verbose=False,
        )
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
    threads = 256
    blocks = int(rows)
    mod.quant_dequant_rowwise(
        (blocks,), (threads,),
        [x32.data_ptr(), scales.data_ptr(), y.data_ptr(), rows, cols, qmax],
        stream=torch.cuda.current_stream().cuda_stream,
    )
    return y.squeeze(0) if squeeze else y


