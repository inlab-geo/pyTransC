"""Common fixtures for testing pseudopriors."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from pytransc.utils.types import FloatArray


@pytest.fixture
def ensemble_per_state() -> list[FloatArray]:
    """Fixture for ensemble per state."""
    mus = [0.0, 1.0]
    covs = [1, 2]
    state_rvs = [multivariate_normal(mean=mu, cov=cov) for mu, cov in zip(mus, covs)]

    return [state_rv.rvs(size=50_000)[..., np.newaxis] for state_rv in state_rvs]
