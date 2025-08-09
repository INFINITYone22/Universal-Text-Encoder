// Copyright (c) 2025 ROHITH GARAPATI (INFINITYone22)
// Universal Text Encoder - CUDA/C++ extensions

#include <torch/extension.h>

// CUDA implementation (defined in .cu)
torch::Tensor qdq_rowwise_cuda(torch::Tensor x, int64_t num_bits);

// CPU fallback: simple per-row symmetric affine q->dq in float
torch::Tensor qdq_rowwise_cpu(torch::Tensor x, int64_t num_bits) {
  TORCH_CHECK(x.dim() == 2, "Input must be 2D (rows, cols)");
  TORCH_CHECK(num_bits == 4 || num_bits == 8, "num_bits must be 4 or 8");
  auto rows = x.size(0);
  auto cols = x.size(1);
  auto x_contig = x.contiguous().to(torch::kFloat32);
  auto y = torch::empty_like(x_contig);

  const int qmax = (1 << (num_bits - 1)) - 1;
  for (int64_t r = 0; r < rows; ++r) {
    auto row = x_contig[r];
    float amax = row.abs().amax().item<float>();
    float scale = std::max(amax / static_cast<float>(qmax), 1e-8f);
    for (int64_t c = 0; c < cols; ++c) {
      float v = row[c].item<float>() / scale;
      v = std::nearbyint(v);
      float lo = -static_cast<float>(qmax) - 1.0f;
      float hi = static_cast<float>(qmax);
      if (v < lo) v = lo;
      if (v > hi) v = hi;
      y[r][c] = v * scale;
    }
  }
  return y;
}

torch::Tensor qdq_rowwise(torch::Tensor x, int64_t num_bits) {
  if (x.is_cuda()) {
    return qdq_rowwise_cuda(x, num_bits);
  }
  return qdq_rowwise_cpu(x, num_bits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "UTE quantize-dequantize rowwise kernels (CUDA/C++)";
  m.def("qdq_rowwise", &qdq_rowwise, "Rowwise quantize-dequantize (CPU/CUDA)",
        py::arg("x"), py::arg("num_bits"));
}


