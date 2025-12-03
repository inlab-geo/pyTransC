"""Tests for the mean-covariance pseudo-prior."""

import numpy as np
import pytest

from pytransc.pseudoprior.mean_covariance import (
    MeanCovariancePseudoPrior,
    build_mean_covariance_pseudo_prior,
)
from pytransc.utils.types import FloatArray


@pytest.fixture
def mean_cov_pseudo(ensemble_per_state: list[FloatArray]) -> MeanCovariancePseudoPrior:
    """Fixture providing a mean-covariance pseudo-prior."""
    return build_mean_covariance_pseudo_prior(ensemble_per_state)


def test_mean_covariance_pseudo_prior_log_density(
    mean_cov_pseudo: MeanCovariancePseudoPrior,
) -> None:
    """Test mean-covariance pseudo-prior log density evaluation."""

    def gaussian_logpdf(x: float, mean: float, cov: float) -> float:
        """Manually compute the log pdf of a univariate Gaussian."""
        return -0.5 * np.log(2 * np.pi * cov) - 0.5 * (x - mean) ** 2 / cov

    for state, rv in enumerate(mean_cov_pseudo.rv_list):
        mu = rv.mean[0]
        cov = rv.cov[0, 0]

        log_density_at_mean = mean_cov_pseudo(np.array([mu]), state)
        assert np.isclose(log_density_at_mean, gaussian_logpdf(mu, mu, cov), atol=0.01)


def test_mean_covariance_pseudo_prior_draw_deviate(
    mean_cov_pseudo: MeanCovariancePseudoPrior,
) -> None:
    """Test mean-covariance pseudo-prior draw deviate."""
    np.random.seed(42)
    for state, rv in enumerate(mean_cov_pseudo.rv_list):
        mu = rv.mean[0]
        cov = rv.cov[0, 0]

        deviates = np.array([mean_cov_pseudo.draw_deviate(state) for _ in range(10_000)])
        assert np.isclose(np.mean(deviates), mu, atol=0.1)
        assert np.isclose(np.var(deviates), cov, atol=0.2)
