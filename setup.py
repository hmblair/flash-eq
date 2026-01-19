"""Build script for flash-eq with optional CUDA extension."""
import os
from setuptools import setup, find_packages

ext_modules = []
cmdclass = {}

# Only build CUDA extension if CUDA is available
try:
    import torch
    if torch.cuda.is_available() or os.environ.get("CUDA_HOME"):
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        ext_modules.append(
            CUDAExtension(
                name="flash_eq._block_diagonal_cuda",
                sources=["flash_eq/cuda/csrc/block_diagonal.cu"],
                extra_compile_args={
                    "cxx": ["-O3"],
                    "nvcc": ["-O3", "--use_fast_math"],
                },
            )
        )
        cmdclass["build_ext"] = BuildExtension
except (ImportError, OSError):
    pass

setup(
    name="flash-eq",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
    package_data={"flash_eq": ["cuda/csrc/*.cu"]},
)
