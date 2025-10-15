"""A helper module for the emcee sampler."""

import inspect
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from emcee import EnsembleSampler

from ..utils.types import FloatArray


def perform_sampling_with_emcee(
    log_prob_func: Callable[[FloatArray], float],
    n_walkers: int,
    n_steps: int,
    initial_state: FloatArray,
    pool: Any | None = None,
    **kwargs,
) -> EnsembleSampler:
    """Perform MCMC sampling using emcee.

    Creates, runs and returns an emcee EnsembleSampler instance.

    Parameters
    ----------
    log_prob_func : Callable[[FloatArray], float]
        Function to evaluate the log-probability density.
    n_walkers : int
        Number of random walkers for the ensemble sampler.
    n_steps : int
        Number of MCMC steps to perform.
    initial_state : FloatArray
        Initial positions for the walkers.
    pool : Any | None, optional
        User-provided pool for parallel processing. If provided, this takes
        precedence over the parallel and n_processors parameters. The pool
        must implement a map() method compatible with the standard library's
        map() function. Default is None.
    **kwargs
        Additional keyword arguments passed to emcee sampler.

    Returns
    -------
    EnsembleSampler
        The emcee sampler instance after running MCMC.

    Examples
    --------
    Using with schwimmbad pools:

    >>> from schwimmbad import MPIPool
    >>> with MPIPool() as pool:
    ...     sampler = perform_sampling_with_emcee(
    ...         log_prob_func=my_log_prob,
    ...         n_walkers=32,
    ...         n_steps=1000,
    ...         initial_state=start_pos,
    ...         pool=pool
    ...     )

    Using with ProcessPoolExecutor:

    >>> from concurrent.futures import ProcessPoolExecutor
    >>> with ProcessPoolExecutor(max_workers=4) as pool:
    ...     sampler = perform_sampling_with_emcee(
    ...         log_prob_func=my_log_prob,
    ...         n_walkers=32,
    ...         n_steps=1000,
    ...         initial_state=start_pos,
    ...         pool=pool
    ...     )
    """

    run_kwargs, initialisation_kwargs, remaining_kwargs = _extract_run_kwargs(kwargs)

    # User-provided pool takes precedence
    if pool is not None:
        initialisation_kwargs["pool"] = pool

    sampler = EnsembleSampler(
        nwalkers=n_walkers,
        ndim=initial_state.shape[1],
        log_prob_fn=log_prob_func,
        **initialisation_kwargs,
    )
    sampler.run_mcmc(initial_state, n_steps, **run_kwargs)

    return sampler


def _extract_run_kwargs(
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Separate kwargs to be passed to the EnsembleSampler initialisation and EnsembleSampler.run_mcmc.

    kwargs for run_mcmc are passed straight to the EnsembleSampler.sample method.
    """
    _run_kwargs = _kwargs_from_instance_method(
        inspect.signature(EnsembleSampler.sample)
    )

    _init_kwargs = _kwargs_from_instance_method(
        inspect.signature(EnsembleSampler.__init__)
    )

    run_kwargs = {k: kwargs.pop(k) for k in _run_kwargs if k in kwargs}
    init_kwargs = {k: kwargs.pop(k) for k in _init_kwargs if k in kwargs}

    return run_kwargs, init_kwargs, kwargs


def _kwargs_from_instance_method(sig: inspect.Signature) -> list[str]:
    """Extract kwargs from an instance method signature.

    Explicitly excludes `self` just in case.
    """
    return [
        name
        for name, param in sig.parameters.items()
        if param.kind
        in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and name != "self"
    ]
