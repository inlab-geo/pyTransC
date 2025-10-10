"""Module for multivariate gaussian pseudo-prior built using scipy."""

import numpy as np
from scipy import stats
from scipy.stats._multivariate import multivariate_normal_frozen

from ..utils.types import FloatArray


class MeanCovariancePseudoPrior:
    """Class for mean and covariance pseudo-prior."""

    def __init__(self, rv_list: list[multivariate_normal_frozen]):
        """
        Initialize the mean and covariance pseudo-prior.

        Parameters
        ----------
        rv_list : list of scipy.stats._multivariate.multivariate_normal_frozen
            List of multivariate fitted normal distributions for each state.
        """
        self.rv_list = rv_list

    def __call__(self, x: FloatArray, state: int) -> float:
        """Evaluate the log pseudo-prior density."""
        rv = self.rv_list[state]
        return rv.logpdf(x)

    def draw_deviate(self, state: int) -> FloatArray:
        """Draw a random deviate from the pseudo-prior for a given state."""
        rv = self.rv_list[state]
        return rv.rvs(size=1)


def build_mean_covariance_pseudo_prior(
    ensemble_per_state: list[FloatArray],
) -> MeanCovariancePseudoPrior:
    """
    Build a mean and covariance pseudo-prior function.

    Args:
        ensemble_per_state (list[FloatArray]): List of ensembles for each state.

    Returns:
        MeanCovariancePseudoPrior: Instance of MeanCovariancePseudoPrior.
    """
    rv_list = []
    for state_ensemble in ensemble_per_state:
        pseudo_covariances = np.cov(state_ensemble.T)
        pseudo_means = np.mean(state_ensemble.T, axis=1)
        rv = stats.multivariate_normal(mean=pseudo_means, cov=pseudo_covariances)
        rv_list.append(rv)
    return MeanCovariancePseudoPrior(rv_list)
