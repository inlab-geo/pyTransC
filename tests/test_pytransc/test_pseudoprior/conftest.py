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


@pytest.fixture
def log_pseudo_prior_fn():
    """Example log pseudo-prior function factory."""
    def _log_pseudo_prior_fn(x: FloatArray, state: int) -> float:
        return state * x.sum()
    return _log_pseudo_prior_fn

@pytest.fixture
def draw_deviate_fn():
    """Example draw deviate function factory."""
    def _draw_deviate_fn(state: int) -> FloatArray:
        return np.ones(state)
    return _draw_deviate_fn
