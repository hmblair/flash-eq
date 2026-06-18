"""Build script for flash-eq with CUDA extension."""
import os
import sys
from setuptools import setup, find_packages

_skip_cuda = os.environ.get("FLASH_EQ_SKIP_CUDA", "").lower() in ("1", "true", "yes")

ext_modules = []
cmdclass = {}

if _skip_cuda:
    print(
        "FLASH_EQ_SKIP_CUDA set; skipping CUDA extension. "
        "The resulting install will fail at runtime when a CUDA kernel is invoked.",
        file=sys.stderr,
    )
else:
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except ImportError as e:
        raise RuntimeError(
            "Cannot build flash-eq: PyTorch is required at build time but could "
            f"not be imported ({e}).\n\n"
            "Install torch first, or use an isolated build (the default for "
            "`pip install`)."
        ) from e

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

setup(
    name="flash-eq",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
    package_data={"flash_eq": ["cuda/csrc/*.cu"]},
)
