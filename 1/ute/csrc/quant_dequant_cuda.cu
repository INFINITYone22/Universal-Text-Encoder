// Copyright (c) 2025 ROHITH GARAPATI (INFINITYone22)
// Universal Text Encoder - CUDA/C++ extensions

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

__global__ void qdq_kernel(const float* __restrict__ x,
                           const float* __restrict__ scales,
                           float* __restrict__ y,
                           long rows, long cols, int qmax) {
  long r = blockIdx.x;
  if (r >= rows) return;
  float scale = scales[r];
  long base = r * cols;
  for (long c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = x[base + c] / scale;
    v = nearbyintf(v);
    float lo = -float(qmax) - 1.0f;
    float hi = float(qmax);
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    y[base + c] = v * scale;
  }
}

} // namespace

torch::Tensor qdq_rowwise_cuda(torch::Tensor x, int64_t num_bits) {
  TORCH_CHECK(x.is_cuda(), "Input must be CUDA tensor");
  TORCH_CHECK(x.dim() == 2, "Input must be 2D (rows, cols)");
  TORCH_CHECK(num_bits == 4 || num_bits == 8, "num_bits must be 4 or 8");

  auto rows = x.size(0);
  auto cols = x.size(1);
  auto x32 = x.contiguous().to(torch::kFloat32);
  auto y = torch::empty_like(x32);

  int qmax = (1 << (num_bits - 1)) - 1;
  // Compute per-row scales on GPU
  auto amax = std::get<0>(x32.abs().amax(1));
  auto scales = amax.clamp_min(1e-8).div(static_cast<float>(qmax));

  int threads = 256;
  int blocks = static_cast<int>(rows);
  qdq_kernel<<<blocks, threads>>>(
      x32.data_ptr<float>(), scales.data_ptr<float>(), y.data_ptr<float>(),
      rows, cols, qmax);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "qdq_kernel launch failed: ", cudaGetErrorString(err));
  return y;
}


