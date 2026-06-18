"""Build script for flash-eq with CUDA extension."""
from setuptools import setup, find_packages

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError as e:
    raise RuntimeError(
        "Cannot build flash-eq: PyTorch is required at build time but could "
        f"not be imported ({e}).\n\n"
        "Install torch first, or use an isolated build (the default for "
        "`pip install`)."
    ) from e

setup(
    name="flash-eq",
    ext_modules=[
        CUDAExtension(
            name="flash_eq._block_diagonal_cuda",
            sources=["flash_eq/cuda/csrc/block_diagonal.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=find_packages(),
    package_data={"flash_eq": ["cuda/csrc/*.cu"]},
)
