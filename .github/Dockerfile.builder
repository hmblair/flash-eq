FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common git && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev \
    python3.11 python3.11-venv python3.11-dev \
    python3.12 python3.12-venv python3.12-dev && \
    rm -rf /var/lib/apt/lists/*

# Pre-install torch CPU and build tools for each Python version
# Install torch with CUDA support (needed for torch.utils.cpp_extension).
# Use --no-deps to skip bundled NVIDIA libs (provided by the base image),
# then install the non-NVIDIA deps manually.
RUN for pyver in 3.10 3.11 3.12; do \
    python$pyver -m venv /opt/venv-$pyver && \
    /opt/venv-$pyver/bin/pip install --no-cache-dir --no-deps \
        torch --index-url https://download.pytorch.org/whl/cu129 && \
    /opt/venv-$pyver/bin/pip install --no-cache-dir \
        filelock jinja2 networkx sympy typing-extensions && \
    /opt/venv-$pyver/bin/pip install --no-cache-dir \
        "setuptools>=61.0" setuptools-scm wheel; \
    done
