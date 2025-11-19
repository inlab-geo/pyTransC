"""Build an automatic pseudo-prior."""

from enum import StrEnum, auto
from typing import Any, Protocol

from ..utils.exceptions import InputError
from ..utils.types import (
    FloatArray,
    SampleableMultiStateDensity,
)
from .gaussian_mixture import build_gaussian_mixture_pseudo_prior
from .mean_covariance import build_mean_covariance_pseudo_prior


class PseudoPriorBuilders(StrEnum):
    """Enum for available pseudo-prior builders."""

    GAUSSIAN_MIXTURE = auto()
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


pseudo_prior_factories: dict[PseudoPriorBuilders, PseudoPriorBuilder] = {
    PseudoPriorBuilders.GAUSSIAN_MIXTURE: build_gaussian_mixture_pseudo_prior,
    PseudoPriorBuilders.MEAN_COVARIANCE: build_mean_covariance_pseudo_prior,
}


def build_auto_pseudo_prior(
    ensemble_per_state: list[FloatArray],
    pseudo_prior_type: PseudoPriorBuilders = PseudoPriorBuilders.GAUSSIAN_MIXTURE,
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
        Type of pseudo-prior builder to use. Default is GAUSSIAN_MIXTURE.
        Options include:
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
    try:
        pseudo_prior_type = PseudoPriorBuilders(pseudo_prior_type)
    except ValueError as e:
        raise InputError(f"Invalid pseudo_prior_type: {pseudo_prior_type}") from e

    log_pseudo_prior = pseudo_prior_factories[pseudo_prior_type](
        ensemble_per_state, **builder_kwargs
    )

    return log_pseudo_prior
