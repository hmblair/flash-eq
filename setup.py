"""Build script for flash-eq with CUDA extension."""
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cuda_extension = CUDAExtension(
    name="flash_eq._block_diagonal_cuda",
    sources=["flash_eq/cuda/csrc/block_diagonal.cu"],
    extra_compile_args={
        "cxx": ["-O3"],
        "nvcc": ["-O3", "--use_fast_math"],
    },
)

setup(
    name="flash-eq",
    ext_modules=[cuda_extension],
    cmdclass={"build_ext": BuildExtension},
    packages=find_packages(),
    package_data={"flash_eq": ["cuda/csrc/*.cu"]},
)
