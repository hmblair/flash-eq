"""Build script for flash-eq with optional CUDA extension."""
import os
from pathlib import Path
from setuptools import setup, find_packages

ext_modules = []
cmdclass = {}

def write_build_info(arch_list: list[str]) -> None:
    """Write build info file with compiled architectures."""
    build_info_path = Path(__file__).parent / "flash_eq" / "_build_info.py"
    build_info_path.write_text(
        f'"""Auto-generated build info."""\n'
        f'COMPILED_ARCHS = {arch_list!r}\n'
    )

# Only build CUDA extension if CUDA is available
try:
    import torch
    if torch.cuda.is_available() or os.environ.get("CUDA_HOME"):
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        # GPU architectures: A100, L40S, H100 + PTX for forward compatibility
        # Users with other GPUs can set TORCH_CUDA_ARCH_LIST or use JIT fallback
        nvcc_args = ["-O3", "--use_fast_math"]
        compiled_archs: list[str] = []

        arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
        if arch_list:
            # Parse "8.0;8.9;9.0" or "8.0 8.9 9.0" format
            for arch in arch_list.replace(" ", ";").split(";"):
                if arch:
                    major, minor = arch.split(".")
                    nvcc_args.append(f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}")
                    compiled_archs.append(f"{major}.{minor}")
        else:
            nvcc_args.extend([
                "-gencode=arch=compute_80,code=sm_80",   # A100
                "-gencode=arch=compute_89,code=sm_89",   # L40S, RTX 4090
                "-gencode=arch=compute_90,code=sm_90",   # H100
                "-gencode=arch=compute_100,code=sm_100", # B100, B200
                "-gencode=arch=compute_120,code=sm_120", # RTX 5090
                "-gencode=arch=compute_120,code=compute_120",  # PTX for future GPUs
            ])
            compiled_archs = ["8.0", "8.9", "9.0", "10.0", "12.0+PTX"]

        write_build_info(compiled_archs)

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
