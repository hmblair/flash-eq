FROM quay.io/pypa/manylinux2014_x86_64

# Install CUDA 12.4 toolkit and git
RUN yum install -y yum-utils git && \
    yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo && \
    yum install -y cuda-toolkit-12-4 && \
    yum clean all

ENV CUDA_HOME=/usr/local/cuda-12.4
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Create venvs with the naming convention the release workflow expects
RUN for pyver in 3.10 3.11 3.12 3.13; do \
    cpver="cp${pyver/./}"; \
    /opt/python/${cpver}-${cpver}/bin/python -m venv /opt/venv-$pyver && \
    /opt/venv-$pyver/bin/pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cu124 && \
    /opt/venv-$pyver/bin/pip install --no-cache-dir \
        "setuptools>=61.0" setuptools-scm wheel; \
    done
