FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common git patchelf && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev \
    python3.11 python3.11-venv python3.11-dev \
    python3.12 python3.12-venv python3.12-dev \
    python3.13 python3.13-venv python3.13-dev && \
    rm -rf /var/lib/apt/lists/*

# Pre-install torch (CUDA-enabled), build tools, and auditwheel
RUN for pyver in 3.10 3.11 3.12 3.13; do \
    python$pyver -m venv /opt/venv-$pyver && \
    /opt/venv-$pyver/bin/pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cu124 && \
    /opt/venv-$pyver/bin/pip install --no-cache-dir \
        "setuptools>=61.0" setuptools-scm wheel auditwheel; \
    done
