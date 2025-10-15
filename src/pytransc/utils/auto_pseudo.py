"""Build an automatic pseudo-prior."""

from enum import StrEnum, auto
from typing import Any, Protocol

import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture

from .types import FloatArray, SampleableMultiStateDensity


class PseudoPriorBuilders(StrEnum):
    """Enum for available pseudo-prior builders."""

    GAUSSIAN_MIXTURE = auto()
    GAUSSIAN_MIXTURE_STANDARDIZED = auto()
    MEAN_COVARIANCE = auto()


class PseudoPriorBuilder(Protocol):
    """Protocol for pseudo-prior builder function."""

    def __call__(
        self,
        ensemble_per_state: list[FloatArray],
        *args: Any,
        **kwargs: Any,
    ) -> SampleableMultiStateDensity:
        """
        Build a pseudo-prior function based on the provided parameters.

        Args:
            ensemble_per_state (list[FloatArray]): List of ensembles for each state.  Each ensemble should be appropriately distributed.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns a callable pseudo-prior function.
        """
        ...


def rescale_gmm_covariances(gmm: GaussianMixture,
                            s: np.ndarray,
                            new_means: np.ndarray = None,
                            new_weights: np.ndarray = None,
                            verbose: bool = False) -> GaussianMixture:
    """
    Rescales the covariance matrices (covariances_) of a GaussianMixture model
    by a known scaling vector 's'. Optionally updates means and weights, ensures
    internal consistency, and returns the updated model.

    The transformation applied to covariance is Sigma' = S @ Sigma @ S, where S
    is the diagonal matrix derived from the scaling vector s.

    Args:
        gmm (GaussianMixture): The scikit-learn GaussianMixture model object.
        s (np.ndarray): The scaling vector (1D array) where s[i] scales the i-th feature.
                        For 'spherical' covariance, s must be a single scalar.
        new_means (np.ndarray, optional): New mean vectors (n_components, n_features).
                                          If provided, gmm.means_ is updated.
        new_weights (np.ndarray, optional): New mixing weights (n_components,). Must
                                            sum to 1.0. If provided, gmm.weights_ is updated.
        verbose (bool, optional): If True, print status and warning messages. Defaults to False.

    Returns:
        GaussianMixture: The updated GaussianMixture model object.

    Raises:
        ValueError: If array shapes or properties (like weight sum) are invalid.
        NotImplementedError: For unsupported covariance types.
    """

    # We assume means_ is already set and use its shape to determine n_features.
    if not hasattr(gmm, 'means_') or gmm.means_ is None:
         raise ValueError(
             "GMM must be initialized or fitted (i.e., gmm.means_ must be set) "
             "before proceeding."
         )

    n_features = gmm.means_.shape[1]
    n_components = gmm.n_components

    # --- 0. Input Validation and Setup for Scaling ---

    if gmm.covariance_type == 'spherical':
        # Spherical covariance requires a scalar (uniform scale across all features)
        # Accept either a 0-d scalar or extract scalar from 1-d array
        if s.ndim == 0:
            s_value = float(s)
        elif s.ndim == 1 and s.size == 1:
            # Single-element array: extract the scalar value
            s_value = float(s[0])
        else:
            raise ValueError(
                f"For 'spherical' covariance_type, the scaling factor 's' must be a scalar "
                f"or single-element array. Provided shape: {s.shape}"
            )
        # For the 'spherical' case, the matrix S is not used.
        S = None
    else:
        # Non-spherical types require a 1D scaling vector with length = n_features
        # Note: For 1D parameter spaces (n_features=1), s.shape=(1,) is correct!
        if s.ndim != 1:
            raise ValueError(
                f"For '{gmm.covariance_type}' covariance_type, scaling vector 's' must be "
                f"1D. Provided shape: {s.shape} with ndim={s.ndim}"
            )
        if s.shape[0] != n_features:
            raise ValueError(
                f"For '{gmm.covariance_type}' covariance_type, scaling vector 's' must have "
                f"length equal to n_features ({n_features}). Provided shape: {s.shape}"
            )
        # Create the diagonal scaling matrix S from the vector s
        S = np.diag(s)


    # --- 1. OPTIONAL: Update Means and Weights ---
    if new_means is not None:
        expected_shape = (n_components, n_features)
        if new_means.shape != expected_shape:
            raise ValueError(f"Shape of new_means must match {expected_shape}. Provided shape: {new_means.shape}")
        gmm.means_ = new_means
        if verbose:
            print("Successfully updated means_.")

    if new_weights is not None:
        expected_shape = (n_components,)
        if new_weights.shape != expected_shape:
            raise ValueError(f"Shape of new_weights must match {expected_shape}. Provided shape: {new_weights.shape}")
        if not np.isclose(np.sum(new_weights), 1.0) or (new_weights < 0).any():
            raise ValueError("New weights must sum close to 1.0 and be non-negative.")
        gmm.weights_ = new_weights
        if verbose:
            print("Successfully updated weights_.")


    # --- 2. MANDATORY: Rescale Covariances ---

    cov_type = gmm.covariance_type

    if verbose:
        print(f"\n--- Rescaling Covariances (Type: '{cov_type}') ---")

    if cov_type == 'full':
        new_covariances = np.empty_like(gmm.covariances_)
        for k in range(gmm.n_components):
            Sigma_k = gmm.covariances_[k]
            new_covariances[k] = S @ Sigma_k @ S # S * Sigma * S

        gmm.covariances_ = new_covariances
        if verbose:
            print("Successfully rescaled 'full' covariances for all components.")

    elif cov_type == 'diag':
        s_squared = s ** 2
        new_covariances = gmm.covariances_ * s_squared # Element-wise multiply variances by s_i^2

        gmm.covariances_ = new_covariances
        if verbose:
            print("Successfully rescaled 'diag' covariances using element-wise squared scaling.")

    elif cov_type == 'spherical':
        s_squared = s_value ** 2
        new_covariances = gmm.covariances_ * s_squared # Element-wise multiply spherical variances by s^2

        gmm.covariances_ = new_covariances
        if verbose:
            print("Successfully rescaled 'spherical' covariances using scalar squared scaling.")

    elif cov_type == 'tied':
        Sigma_tied = gmm.covariances_
        new_covariances = S @ Sigma_tied @ S

        gmm.covariances_ = new_covariances
        if verbose:
            print("Successfully rescaled 'tied' covariance matrix.")

    else:
        raise NotImplementedError(
            f"Unsupported covariance_type: '{cov_type}'"
        )

    # --- 3. CRITICAL FIX: Recompute precisions_cholesky_ ---
    # REQUIRED: Uses the correct attribute name 'precisions_cholesky_'
    try:
        if gmm.covariance_type in ('full', 'tied'):
            # For full or tied, we compute the Cholesky decomposition of the precision matrix (inverse covariance)

            # Make covs 3D (even if tied) for consistent looping
            if gmm.covariance_type == 'full':
                covs = gmm.covariances_
            else: # 'tied'
                covs = gmm.covariances_[np.newaxis, :, :]

            new_precision_cholesky = np.empty(covs.shape)
            for k in range(covs.shape[0]):
                # 1. Calculate precision (inverse covariance)
                precision = np.linalg.inv(covs[k])
                # 2. Calculate Cholesky decomposition of the precision matrix (L, such that L @ L.T = precision)
                # The result is stored as the transpose (lower triangular) in scikit-learn convention.
                new_precision_cholesky[k] = np.linalg.cholesky(precision)

            # For 'tied', remove the added first dimension; for 'full', keep all dimensions
            if gmm.covariance_type == 'tied':
                gmm.precisions_cholesky_ = new_precision_cholesky[0]  # Remove first dimension only
            else:
                gmm.precisions_cholesky_ = new_precision_cholesky  # Keep shape (n_components, n_features, n_features)

        elif gmm.covariance_type in ('diag', 'spherical'):
            # For diag and spherical, Cholesky of precision is 1 / sqrt(variance)
            gmm.precisions_cholesky_ = 1. / np.sqrt(gmm.covariances_)

        if verbose:
            print("Successfully recomputed precisions_cholesky_ using direct NumPy operations.")

    except np.linalg.LinAlgError:
        print("Warning: Could not recompute precisions_cholesky_ because the rescaled "
              "covariance matrix is not positive definite. score_samples will fail.")
    except Exception as e:
        print(f"Warning: Could not recompute precisions_cholesky_ due to an unexpected error. Error: {e}")


    return gmm # Explicitly return the modified GMM object


class GaussianMixturePseudoPrior:
    """Class for Gaussian mixture pseudo-prior."""

    def __init__(self, ensemble_per_state: list[FloatArray], **kwargs):
        """
        Initialize the Gaussian mixture pseudo-prior.

        Parameters
        ----------
        ensemble_per_state : list of FloatArray
            List of ensembles for each state.
        kwargs : dict
            Additional arguments for Gaussian mixture fitting.
        """
        self.gaussian_mixtures = [
            GaussianMixture(**kwargs).fit(ens) for ens in ensemble_per_state
        ]

    def __call__(self, x: FloatArray, state: int) -> float:
        """Evaluate the log pseudo-prior density."""
        gmm = self.gaussian_mixtures[state]
        return float(gmm.score([x]))

    def draw_deviate(self, state: int) -> FloatArray:
        """Draw a random deviate from the pseudo-prior for a given state."""
        gmm = self.gaussian_mixtures[state]
        return gmm.sample()[0][0]


class MeanCovariancePseudoPrior:
    """Class for mean and covariance pseudo-prior."""

    def __init__(self, ensemble_per_state: list[FloatArray]):
        """
        Initialize the mean and covariance pseudo-prior.

        Parameters
        ----------
        ensemble_per_state : list of FloatArray
            List of ensembles for each state.
        """
        self.rv_list = []
        for state_ensemble in ensemble_per_state:
            pseudo_covariances_ = np.cov(state_ensemble.T)
            pseudo_means_ = np.mean(state_ensemble.T, axis=1)
            rv = stats.multivariate_normal(mean=pseudo_means_, cov=pseudo_covariances_)
            self.rv_list.append(rv)

    def __call__(self, x: FloatArray, state: int) -> float:
        """Evaluate the log pseudo-prior density."""
        rv = self.rv_list[state]
        return rv.logpdf(x)

    def draw_deviate(self, state: int) -> FloatArray:
        """Draw a random deviate from the pseudo-prior for a given state."""
        rv = self.rv_list[state]
        return rv.rvs(size=1)[0]


class GaussianMixtureStandardizedPseudoPrior:
    """Class for Gaussian mixture pseudo-prior with standardization.

    This class standardizes the ensemble before fitting the Gaussian mixture,
    then transforms the fitted mixture back to the original scale. This improves
    GMM fitting when parameters have different scales.
    """

    def __init__(self, ensemble_per_state: list[FloatArray], **kwargs):
        """
        Initialize the standardized Gaussian mixture pseudo-prior.

        Parameters
        ----------
        ensemble_per_state : list of FloatArray
            List of ensembles for each state.
        kwargs : dict
            Additional arguments for Gaussian mixture fitting.
        """
        self.gaussian_mixtures = []
        for ens in ensemble_per_state:
            # Compute mean and std of ensemble
            mean = np.mean(ens, axis=0)
            std = np.std(ens, axis=0)

            # Standardize ensemble
            ens_stan = (ens - mean) / std

            # Fit GMM to standardized data
            gmm = GaussianMixture(**kwargs).fit(ens_stan)

            # Transform means back to original scale
            newmeans = std * gmm.means_ + mean

            # Special handling for spherical covariance
            if gmm.covariance_type == 'spherical':
                std = np.mean(std)

            # Rescale covariances back to original scale
            gmm = rescale_gmm_covariances(gmm, std, newmeans, gmm.weights_)

            self.gaussian_mixtures.append(gmm)

    def __call__(self, x: FloatArray, state: int) -> float:
        """Evaluate the log pseudo-prior density."""
        gmm = self.gaussian_mixtures[state]
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return gmm.score_samples(x)

    def draw_deviate(self, state: int) -> FloatArray:
        """Draw a random deviate from the pseudo-prior for a given state."""
        gmm = self.gaussian_mixtures[state]
        return gmm.sample()[0][0]


pseudo_prior_factories: dict[PseudoPriorBuilders, PseudoPriorBuilder] = {
    PseudoPriorBuilders.GAUSSIAN_MIXTURE: GaussianMixturePseudoPrior,
    PseudoPriorBuilders.GAUSSIAN_MIXTURE_STANDARDIZED: GaussianMixtureStandardizedPseudoPrior,
    PseudoPriorBuilders.MEAN_COVARIANCE: MeanCovariancePseudoPrior,
}


def build_auto_pseudo_prior(
    ensemble_per_state: list[FloatArray],
    pseudo_prior_type: PseudoPriorBuilders = PseudoPriorBuilders.GAUSSIAN_MIXTURE_STANDARDIZED,
    **builder_kwargs,
):
    """
    Build an automatic pseudo-prior function using a specified builder.

    This function fits a statistical model (e.g., Gaussian mixture) to posterior
    ensembles from each state to create a pseudo-prior for trans-conceptual sampling.

    Parameters
    ----------
    ensemble_per_state : list of FloatArray
        List of posterior samples for each state. Each array should have shape
        (n_samples, n_features) where n_features is the dimensionality of that state.
        Generate these samples using run_mcmc_per_state() or provide pre-existing
        posterior ensembles.
    pseudo_prior_type : PseudoPriorBuilders, optional
        Type of pseudo-prior builder to use. Default is GAUSSIAN_MIXTURE_STANDARDIZED.
        Options include:
        - GAUSSIAN_MIXTURE_STANDARDIZED: GMM with standardization (recommended)
        - GAUSSIAN_MIXTURE: Standard GMM without standardization
        - MEAN_COVARIANCE: Simple Gaussian approximation
    **builder_kwargs : dict
        Additional arguments passed to the pseudo-prior builder (e.g., n_components,
        covariance_type for Gaussian mixture models).

    Returns
    -------
    log_pseudo_prior : SampleableMultiStateDensity
        Callable pseudo-prior object with methods:
        - __call__(x, state): Evaluate log pseudo-prior at point x in given state
        - draw_deviate(state): Sample from pseudo-prior for given state

    Examples
    --------
    >>> # Generate posterior ensembles for each state
    >>> ensemble_per_state, _ = run_mcmc_per_state(
    ...     n_states=3, n_dims=[2, 3, 4], n_walkers=32, n_steps=5000,
    ...     pos=initial_positions, log_posterior=log_post_func
    ... )
    >>>
    >>> # Build pseudo-prior from ensembles
    >>> pseudo_prior = build_auto_pseudo_prior(
    ...     ensemble_per_state=ensemble_per_state,
    ...     pseudo_prior_type=PseudoPriorBuilders.GAUSSIAN_MIXTURE_STANDARDIZED,
    ...     n_components=3, covariance_type='full'
    ... )
    >>>
    >>> # Use in trans-conceptual sampler
    >>> result = run_state_jump_sampler(..., log_pseudo_prior=pseudo_prior)

    Notes
    -----
    This function no longer supports automatic ensemble generation. If you previously
    used log_posterior and sampling_args parameters, please refactor to:

    1. Generate ensembles explicitly using run_mcmc_per_state()
    2. Pass the resulting ensemble_per_state to this function

    This separation provides better control over sampling parameters and makes the
    workflow more transparent.
    """
    if ensemble_per_state is None:
        raise ValueError(
            "ensemble_per_state is required and cannot be None.\n\n"
            "The automatic ensemble generation feature has been removed for clarity.\n"
            "Please generate posterior ensembles explicitly using run_mcmc_per_state():\n\n"
            "  ensemble_per_state, _ = run_mcmc_per_state(\n"
            "      n_states=n_states,\n"
            "      n_dims=n_dims,\n"
            "      n_walkers=n_walkers,\n"
            "      n_steps=n_steps,\n"
            "      pos=initial_positions,\n"
            "      log_posterior=log_posterior_func,\n"
            "      auto_thin=True\n"
            "  )\n\n"
            "Then pass the ensembles to build_auto_pseudo_prior():\n\n"
            "  log_pseudo_prior = build_auto_pseudo_prior(\n"
            "      ensemble_per_state=ensemble_per_state\n"
            "  )"
        )

    log_pseudo_prior = pseudo_prior_factories[pseudo_prior_type](
        ensemble_per_state, **builder_kwargs
    )

    return log_pseudo_prior
