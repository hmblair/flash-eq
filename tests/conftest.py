"""Shared pytest fixtures and configuration for flash-eq tests."""

import pytest
import torch


def get_available_devices() -> list[torch.device]:
    """Return list of available devices for testing."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


AVAILABLE_DEVICES = get_available_devices()
DEVICE_IDS = [str(d) for d in AVAILABLE_DEVICES]


@pytest.fixture(params=AVAILABLE_DEVICES, ids=DEVICE_IDS)
def device(request) -> torch.device:
    """Parametrized fixture that runs tests on all available devices."""
    return request.param


@pytest.fixture
def cpu_device() -> torch.device:
    """Fixture for CPU-only tests."""
    return torch.device("cpu")


@pytest.fixture
def cuda_device() -> torch.device:
    """Fixture for CUDA-only tests. Skips if CUDA unavailable."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_cuda: mark test as requiring CUDA (skipped if unavailable)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests marked with requires_cuda when CUDA is unavailable."""
    if torch.cuda.is_available():
        return

    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "requires_cuda" in item.keywords:
            item.add_marker(skip_cuda)
