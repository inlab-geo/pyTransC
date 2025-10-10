"""Tests for the mean-covariance pseudo-prior."""

import numpy as np
from scipy.stats import multivariate_normal

from pytransc.pseudoprior.mean_covariance import build_mean_covariance_pseudo_prior


def gaussian_logpdf(x: float, mean: float, cov: float) -> float:
    """Manually compute the log pdf of a univariate Gaussian."""
    return -0.5 * np.log(2 * np.pi * cov) - 0.5 * (x - mean) ** 2 / cov


def test_mean_covariance_pseudo_prior():
    """Test mean-covariance pseudo-prior."""
    mus = [0.0, 1.0]
    covs = [1, 2]
    state_rvs = [multivariate_normal(mean=mu, cov=cov) for mu, cov in zip(mus, covs)]

    ensemble_per_state = [
        state_rv.rvs(size=50_000)[..., np.newaxis] for state_rv in state_rvs
    ]

    mean_cov_pseudo = build_mean_covariance_pseudo_prior(ensemble_per_state)

    for state, (mu, cov) in enumerate(zip(mus, covs)):
        log_density_at_mean = mean_cov_pseudo(np.array([mu]), state)
        assert np.isclose(log_density_at_mean, gaussian_logpdf(mu, mu, cov), atol=0.01)

        deviates = np.array([mean_cov_pseudo.draw_deviate(state) for _ in range(1_000)])
        assert np.isclose(np.mean(deviates), mu, atol=0.1)
        assert np.isclose(np.var(deviates), cov, atol=0.1)
