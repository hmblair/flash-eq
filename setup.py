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

        # Support multiple GPU architectures:
        # - A100: sm_80 (Ampere)
        # - L40S: sm_89 (Ada Lovelace)
        # - H100: sm_90 (Hopper)
        nvcc_args = [
            "-O3",
            "--use_fast_math",
            "-gencode=arch=compute_80,code=sm_80",  # A100
            "-gencode=arch=compute_89,code=sm_89",  # L40S
            "-gencode=arch=compute_90,code=sm_90",  # H100
        ]

        ext_modules.append(
            CUDAExtension(
                name="flash_eq._block_diagonal_cuda",
                sources=["flash_eq/cuda/csrc/block_diagonal.cu"],
                extra_compile_args={
                    "cxx": ["-O3"],
                    "nvcc": nvcc_args,
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
