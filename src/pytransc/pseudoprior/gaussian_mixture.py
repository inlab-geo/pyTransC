"""Module for Gaussian mixture pseudo-prior built using sklearn."""

from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture
from traitlets import Float

from ..utils.types import FloatArray


class StandardGaussianMixture(GaussianMixture):
    """Wrapper for GaussianMixture that standardizes samples before fitting."""

    def fit(self, X: FloatArray, y: Any = None) -> "StandardGaussianMixture":
        """Fit the Gaussian mixture to standardized data."""
        X_std = self._standardize_samples(X)
        return super().fit(X_std)

    def sample(self, n_samples: int = 1) -> tuple[FloatArray, FloatArray]:
        """Draw samples and rescale them back to original scale."""
        samples, labels = super().sample(n_samples=n_samples)
        return samples * self.std + self.mean, labels

    def score_samples(self, X: FloatArray) -> Float:
        """Evaluate the log density of the samples after standardizing them."""
        X_std = self._standardize_samples(X)
        log_density_correction = np.sum(np.log(self.std))
        return super().score_samples(X_std) - log_density_correction

    def _standardize_samples(self, X: FloatArray) -> FloatArray:
        """Standardize samples."""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return (X - self.mean) / self.std


class GaussianMixturePseudoPrior:
    """Class for Gaussian mixture pseudo-prior."""

    def __init__(self, gaussian_mixtures: list[GaussianMixture]) -> None:
        """
        Initialize the Gaussian mixture pseudo-prior.

        Parameters
        ----------
        gaussian_mixtures : list of GaussianMixture
            List of fitted GaussianMixture models, one for each state.
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
        GaussianMixturePseudoPrior: Pseudo-prior object wrapping fitted Gaussian mixture models for each state.
    """
    standardize = kwargs.pop("standardize", False)
    cls = StandardGaussianMixture if standardize else GaussianMixture

    gms = []
    for ens in ensemble_per_state:
        gmm = cls(**kwargs)
        gmm.fit(ens)
        gms.append(gmm)
    return GaussianMixturePseudoPrior(gms)
