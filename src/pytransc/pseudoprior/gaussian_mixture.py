"""Module for Gaussian mixture pseudo-prior built using sklearn."""

from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture

from ..utils.types import FloatArray


class GaussianMixturePseudoPrior:
    """Class for Gaussian mixture pseudo-prior."""

    def __init__(self, gaussian_mixtures: list[GaussianMixture]) -> None:
        """
        Initialize the Gaussian mixture pseudo-prior.

        Parameters
        ----------
        ensemble_per_state : list of FloatArray
            List of ensembles for each state.
        kwargs : dict
            Additional arguments for Gaussian mixture fitting.
        """
        self.gaussian_mixtures = gaussian_mixtures

    def __call__(self, x: FloatArray, state: int) -> float:
        """Evaluate the log pseudo-prior density."""
        gmm = self.gaussian_mixtures[state]
        return float(gmm.score(np.array([x])))

    def draw_deviate(self, state: int) -> FloatArray:
        """Draw a random deviate from the pseudo-prior for a given state."""
        gmm = self.gaussian_mixtures[state]
        return gmm.sample()[0][0]


def build_gaussian_mixture_pseudo_prior(
    ensemble_per_state: list[FloatArray], **kwargs: Any
) -> GaussianMixturePseudoPrior:
    """
    Build a Gaussian mixture pseudo-prior function.

    Args:
        ensemble_per_state (list[FloatArray]): List of ensembles for each state.
        **kwargs: Additional keyword arguments for Gaussian mixture fitting.

    Returns:
        list[GaussianMixture]: List of fitted Gaussian mixture models for each state.
    """
    gms = [
        GaussianMixture(**kwargs).fit(state_ensemble)
        for state_ensemble in ensemble_per_state
    ]
    return GaussianMixturePseudoPrior(gms)
