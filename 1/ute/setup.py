from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="ute_qdq",
    ext_modules=[
        CUDAExtension(
            name="ute_qdq",
            sources=[
                "csrc/quant_dequant_binding.cpp",
                "csrc/quant_dequant_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)


