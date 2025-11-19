"""Tests for the Gaussian Mixture pseudo-prior."""

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

from pytransc.pseudoprior.gaussian_mixture import (
    GaussianMixturePseudoPrior,
    build_gaussian_mixture_pseudo_prior,
)
from pytransc.utils.types import FloatArray


@pytest.fixture
def gm_pseudo(ensemble_per_state: list[FloatArray]) -> GaussianMixturePseudoPrior:
    """Fixture for Gaussian Mixture pseudo-prior."""
    return build_gaussian_mixture_pseudo_prior(ensemble_per_state)


def test_kwargs_handling(ensemble_per_state: list[FloatArray]) -> None:
    """Test that kwargs are correctly passed to GaussianMixture."""
    with pytest.raises(TypeError):
        # n_components should be an integer
        build_gaussian_mixture_pseudo_prior(ensemble_per_state, n_components="invalid")

    with pytest.raises(ValueError):
        # n_components should be positive
        build_gaussian_mixture_pseudo_prior(ensemble_per_state, n_components=0)

    gm_pseudo_prior = build_gaussian_mixture_pseudo_prior(
        ensemble_per_state, n_components=2, reg_covar=1
    )
    for gmm in gm_pseudo_prior.gaussian_mixtures:
        assert gmm.n_components == 2
        assert gmm.reg_covar == 1


def test_gaussian_mixture_pseudo_prior(ensemble_per_state: list[FloatArray]) -> None:
    """Test Gaussian Mixture pseudo-priors are properly fitted."""

    gm_pseudo = build_gaussian_mixture_pseudo_prior(ensemble_per_state)
    for gmm, ensemble in zip(gm_pseudo.gaussian_mixtures, ensemble_per_state):
        assert isinstance(gmm, GaussianMixture)
        assert np.isclose(gmm.means_[0], np.mean(ensemble))
        assert np.isclose(gmm.covariances_[0], np.var(ensemble))


def test_gaussian_mixture_pseudo_prior_log_density(
    gm_pseudo: GaussianMixturePseudoPrior,
) -> None:
    """Test the log density evaluation of Gaussian Mixture pseudo-prior."""

    def gaussian_logpdf(x: float, mean: float, cov: float) -> float:
        """Manually compute the log pdf of a univariate Gaussian."""
        return -0.5 * np.log(2 * np.pi * cov) - 0.5 * (x - mean) ** 2 / cov

    for state, gmm in enumerate(gm_pseudo.gaussian_mixtures):
        log_density_at_mean = gm_pseudo(gmm.means_[0], state)
        assert np.isclose(
            log_density_at_mean,
            gaussian_logpdf(gmm.means_[0], gmm.means_[0], gmm.covariances_[0]),
        )


def test_gaussian_mixture_pseudo_prior_deviate(
    gm_pseudo: GaussianMixturePseudoPrior,
) -> None:
    """Test the sampling from the Gaussian Mixture pseudo-prior."""
    for state, gmm in enumerate(gm_pseudo.gaussian_mixtures):
        deviates = np.array([gm_pseudo.draw_deviate(state) for _ in range(1_000)])
        assert np.isclose(np.mean(deviates), gmm.means_[0], atol=0.1)
        assert np.isclose(np.var(deviates), gmm.covariances_[0], atol=0.1)
