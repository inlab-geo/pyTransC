"""Root conftest.py: session-wide fixtures and configuration."""

import numpy as np
import pytest


# ============================================================================
# Session-wide fixtures
# ============================================================================


@pytest.fixture(scope="session")
def global_random_seed():
    """Global random seed for reproducibility across all tests."""
    return 42


@pytest.fixture(scope="module")
def default_rng(global_random_seed):
    """Default NumPy random number generator for tests."""
    return np.random.default_rng(global_random_seed)


# ============================================================================
# Pytest hooks
# ============================================================================


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on directory location.

    This automatically applies markers based on test location:
    - tests/unit/ -> @pytest.mark.unit
    - tests/integration/ -> @pytest.mark.integration
    - tests/numerical/ -> @pytest.mark.numerical

    Note: Markers must be registered in pyproject.toml [tool.pytest.ini_options]
    """
    for item in items:
        test_path = str(item.fspath)

        # Auto-mark by directory
        if "/unit/" in test_path:
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in test_path:
            item.add_marker(pytest.mark.integration)
        elif "/numerical/" in test_path:
            item.add_marker(pytest.mark.numerical)
