"""Build an automatic pseudo-prior."""

from enum import StrEnum, auto
from typing import Any, Protocol

from ..samplers.per_state import run_mcmc_per_state
from ..utils.exceptions import InputError
from ..utils.types import (
    FloatArray,
    MultiStateDensity,
    MultiStateDraw,
    SampleableMultiStateDensity,
)
from .custom import build_custom_pseudo_prior
from .gaussian_mixture import build_gaussian_mixture_pseudo_prior
from .mean_covariance import build_mean_covariance_pseudo_prior


class PseudoPriorOptions(StrEnum):
    """Enum for available pseudo-prior builders."""

    GAUSSIAN_MIXTURE = auto()
    MEAN_COVARIANCE = auto()
    CUSTOM = auto()


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


pseudo_prior_factories: dict[PseudoPriorOptions, PseudoPriorBuilder] = {
    PseudoPriorOptions.GAUSSIAN_MIXTURE: build_gaussian_mixture_pseudo_prior,
    PseudoPriorOptions.MEAN_COVARIANCE: build_mean_covariance_pseudo_prior,
    PseudoPriorOptions.CUSTOM: build_custom_pseudo_prior,
}


def build_auto_pseudo_prior(
    pseudo_prior_type: PseudoPriorOptions = PseudoPriorOptions.GAUSSIAN_MIXTURE,
    *,
    ensemble_per_state: list[FloatArray] | None = None,
    log_posterior_fn: MultiStateDensity | None = None,
    log_pseudo_prior_fn: MultiStateDensity | None = None,
    draw_deviate_fn: MultiStateDraw | None = None,
    sampling_args: dict[str, Any] | None = None,
    **builder_kwargs,
) -> SampleableMultiStateDensity:
    """
    Build an automatic pseudo-prior function using a specified builder.

    Parameters
    ----------
    pseudo_prior_type : PseudoPriorBuilders, optional
        Type of pseudo-prior builder to use. Default is GAUSSIAN_MIXTURE.
    ensemble_per_state : list of FloatArray, optional
        List of posterior samples for each state. If not provided, samples will be generated using MCMC.
    log_posterior : MultiStateDensity, optional
        Function evaluating the log-posterior for each state. Required if ensemble_per_state is not provided.
    log_pseudo_prior_fn : MultiStateDensity, optional
        Custom function to evaluate the log pseudo-prior density. Required if pseudo_prior_type is CUSTOM.
    draw_deviate_fn : MultiStateDraw, optional
        Custom function to draw a random deviate from the pseudo-prior. Required if pseudo_prior_type is CUSTOM.
    sampling_args : dict, optional
        Arguments for MCMC sampling if ensemble_per_state is not provided.  Must include 'n_states', 'n_dims', 'n_walkers', 'n_steps', and 'pos'.
    **builder_kwargs : dict
        Additional arguments passed to the pseudo-prior builder.

    Returns
    -------
    log_pseudo_prior : PseudoPrior
        Callable function to evaluate the log pseudo-prior at a given point and state.
    """
    try:
        pseudo_prior_type = PseudoPriorOptions(pseudo_prior_type)
    except ValueError:
        raise InputError(f"Invalid pseudo_prior_type: {pseudo_prior_type}")

    builder = pseudo_prior_factories[pseudo_prior_type]

    if ensemble_per_state is None:
        ensemble_per_state = []
    if sampling_args is None:
        sampling_args = {}

    if pseudo_prior_type == PseudoPriorOptions.CUSTOM:
        if log_pseudo_prior_fn is None or draw_deviate_fn is None:
            raise InputError(
                "For CUSTOM pseudo-prior, both log_pseudo_prior_fn and draw_deviate_fn must be provided."
            )
        return builder(
            ensemble_per_state,  # not used in custom builder
            log_pseudo_prior_fn=log_pseudo_prior_fn,
            draw_deviate_fn=draw_deviate_fn,
        )

    if not ensemble_per_state:
        if log_posterior_fn is None:
            raise InputError(
                "log_posterior_fn must be provided if ensemble_per_state is not supplied."
            )
        ensemble_per_state = _generate_missing_ensembles(
            log_posterior=log_posterior_fn, sampling_args=sampling_args
        )

    log_pseudo_prior = builder(ensemble_per_state, **builder_kwargs)

    return log_pseudo_prior


def _generate_missing_ensembles(
    log_posterior: MultiStateDensity,
    sampling_args: dict[str, Any],
) -> list[FloatArray]:
    if any(
        k not in sampling_args
        for k in ["n_states", "n_dims", "n_walkers", "n_steps", "pos"]
    ):
        raise InputError(
            "sampling_args must contain 'n_states', 'n_dims', 'n_walkers', 'n_steps', and 'pos' keys."
        )

    n_states = sampling_args.pop("n_states")
    n_dims = sampling_args.pop("n_dims")
    n_walkers = sampling_args.pop("n_walkers")
    n_steps = sampling_args.pop("n_steps")
    pos = sampling_args.pop("pos")
    return run_mcmc_per_state(
        n_states=n_states,
        n_dims=n_dims,
        n_walkers=n_walkers,
        n_steps=n_steps,
        pos=pos,
        log_posterior=log_posterior,
        **sampling_args,
    )[0]  # return only the ensembles
