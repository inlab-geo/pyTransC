"""Per-State MCMC Sampling."""

import multiprocessing
import random
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any

import numpy as np

from ..utils.types import FloatArray, MultiStateDensity
from ._emcee import perform_sampling_with_emcee


# Global process-level forward pool cache to prevent thrashing
_process_forward_pool = None
_process_forward_pool_config = None

class LogPosteriorWithForwardPool:
    """Picklable wrapper for log_posterior that manages forward pool.
    
    This class allows the log_posterior function to be pickled for use
    with emcee's pool (walker-level parallelism) while still supporting
    forward pool functionality.
    
    Uses a process-level pool cache to prevent pool creation/destruction thrashing.
    """
    
    def __init__(self, log_posterior, forward_pool_config=None):
        """Initialize the wrapper.
        
        Parameters
        ----------
        log_posterior : callable
            The original log posterior function.
        forward_pool_config : dict or None
            Configuration for creating forward pool in worker processes.
        """
        self.log_posterior = log_posterior
        self.forward_pool_config = forward_pool_config
        
    def __call__(self, params):
        """Evaluate log posterior with forward pool context."""
        from ..utils.forward_context import set_forward_pool, clear_forward_pool
        import os
        global _process_forward_pool, _process_forward_pool_config
        
        # Create forward pool at process level if needed (prevents thrashing)
        if (self.forward_pool_config is not None and 
            (_process_forward_pool is None or _process_forward_pool_config != self.forward_pool_config)):
            
            # Clean up old pool if config changed
            if _process_forward_pool is not None and _process_forward_pool_config != self.forward_pool_config:
                _process_forward_pool.shutdown(wait=True)
                _process_forward_pool = None
            
            pool_type = self.forward_pool_config['type']
            pool_kwargs = self.forward_pool_config.get('kwargs', {})
            
            if pool_type == 'ProcessPoolExecutor':
                from concurrent.futures import ProcessPoolExecutor
                _process_forward_pool = ProcessPoolExecutor(**pool_kwargs)
            elif pool_type == 'ThreadPoolExecutor':
                from concurrent.futures import ThreadPoolExecutor
                _process_forward_pool = ThreadPoolExecutor(**pool_kwargs)
                
            _process_forward_pool_config = self.forward_pool_config.copy()
        
        # Set forward pool in context if available
        if _process_forward_pool is not None:
            set_forward_pool(_process_forward_pool)
            try:
                result = self.log_posterior(params)
            finally:
                clear_forward_pool()
        else:
            result = self.log_posterior(params)
            
        return result
    
    def __del__(self):
        """Clean up forward pool on deletion."""
        # No longer managing instance-level pools - cleanup handled at process level
        pass
    
    def cleanup(self):
        """Explicitly clean up the forward pool."""
        global _process_forward_pool, _process_forward_pool_config
        if _process_forward_pool is not None:
            _process_forward_pool.shutdown(wait=True)
            _process_forward_pool = None
            _process_forward_pool_config = None


def run_mcmc_per_state(
    n_states: int,
    n_dims: list[int],
    n_walkers: int | list[int],
    n_steps: int | list[int],
    pos: list[FloatArray],
    log_posterior: MultiStateDensity,
    discard: int | list[int] = 0,
    thin: int | list[int] = 1,
    auto_thin: bool = False,
    seed: int = 61254557,
    state_pool: Any | None = None,
    emcee_pool: Any | None = None,
    n_state_processors: int | None = None,
    skip_initial_state_check: bool = False,
    verbose: bool = True,
    forward_pool: Any | None = None,
    **kwargs,
) -> tuple[list[FloatArray], list[FloatArray]]:
    """Run independent MCMC sampling within each state.

    This utility function runs the emcee sampler independently within each state
    to generate posterior ensembles. These ensembles can then be used as input
    for ensemble resampling or for constructing pseudo-priors.

    Parameters
    ----------
    n_states : int
        Number of independent states in the problem.
    n_dims : list of int
        List of parameter dimensions for each state.
    n_walkers : int or list of int
        Number of random walkers for the emcee sampler. If int, same number
        is used for all states. If list, specifies walkers per state.
    n_steps : int or list of int
        Number of MCMC steps per walker. If int, same number is used for all
        states. If list, specifies steps per state.
    pos : list of FloatArray
        Starting positions for each state. Each array should have shape
        (n_walkers[state], n_dims[state]).
    log_posterior : MultiStateDensity
        Function to evaluate the log-posterior density at location x in state i.
        Must have signature log_posterior(x, state) -> float.
    discard : int or list of int, optional
        Number of samples to discard as burn-in. If int, same value used for
        all states. Default is 0.
    thin : int or list of int, optional
        Thinning factor for chains. If int, same value used for all states.
        Default is 1 (no thinning).
    auto_thin : bool, optional
        If True, automatically thin chains based on autocorrelation time,
        ignoring the `thin` parameter. Default is False.
    seed : int, optional
        Random number seed for reproducible results. Default is 61254557.
    skip_initial_state_check : bool, optional
        Whether to skip emcee's initial state check. Default is False.
    verbose : bool, optional
        Whether to print progress information. Default is True.
    state_pool : Any | None, optional
        User-provided pool for parallelizing state execution. If provided, states
        are processed in parallel. The pool must implement a map() method compatible
        with the standard library's map() function. Default is None.
    emcee_pool : Any | None, optional
        User-provided pool for parallelizing emcee walker execution within each state.
        The pool must implement a map() method compatible with the standard library's
        map() function. Default is None.
    forward_pool : Any | None, optional
        User-provided pool for parallelizing forward solver calls within
        log_posterior evaluations. If provided, the pool will be made available
        to log_posterior functions via get_forward_pool() from pytransc.utils.forward_context.
        The pool must implement a map() method compatible with the standard library's 
        map() function. Supports ProcessPoolExecutor, ThreadPoolExecutor, 
        and schwimmbad pools. Default is None.
    **kwargs
        Additional keyword arguments passed to emcee.EnsembleSampler.

    Returns
    -------
    -------
    ensemble_per_state : list of FloatArray
        Posterior samples for each state. Each array has shape
        (n_samples, n_dims[state]).
    log_posterior_ens : list of FloatArray
        Log posterior values for each ensemble. Each array has shape (n_samples,)

    Notes
    -----
    This function is primarily a convenience wrapper around emcee for generating
    posterior ensembles within each state independently. The resulting ensembles
    can be used with:

    - `run_ensemble_resampler()` for ensemble-based trans-dimensional sampling
    - Automatic pseudo-prior construction functions
    - Direct analysis of within-state posterior distributions

    If `auto_thin=True`, the function will automatically determine appropriate
    burn-in and thinning based on the autocorrelation time, following emcee
    best practices.

    Examples
    --------
    Basic usage:

    >>> ensembles, log_probs = run_mcmc_per_state(
    ...     n_states=2,
    ...     n_dims=[3, 2],
    ...     n_walkers=32,
    ...     n_steps=1000,
    ...     pos=[np.random.randn(32, 3), np.random.randn(32, 2)],
    ...     log_posterior=my_log_posterior,
    ...     auto_thin=True
    ... )

    Using with state-level parallelism:

    >>> from concurrent.futures import ProcessPoolExecutor
    >>> with ProcessPoolExecutor(max_workers=4) as state_pool:
    ...     ensembles, log_probs = run_mcmc_per_state(
    ...         n_states=4,
    ...         n_dims=[3, 2, 4, 1],
    ...         n_walkers=32,
    ...         n_steps=1000,
    ...         pos=initial_positions,
    ...         log_posterior=my_log_posterior,
    ...         state_pool=state_pool
    ...     )
    
    Using with both state and emcee parallelism:
    
    >>> from schwimmbad import MPIPool
    >>> with MPIPool() as state_pool, ProcessPoolExecutor() as emcee_pool:
    ...     ensembles, log_probs = run_mcmc_per_state(
    ...         n_states=2,
    ...         n_dims=[3, 2],
    ...         n_walkers=32,
    ...         n_steps=1000,
    ...         pos=initial_positions,
    ...         log_posterior=my_log_posterior,
    ...         state_pool=state_pool,
    ...         emcee_pool=emcee_pool
    ...     )

    Using with forward pool for parallel forward solver calls:

    >>> from concurrent.futures import ProcessPoolExecutor
    >>> with ProcessPoolExecutor(max_workers=4) as forward_pool:
    ...     ensembles, log_probs = run_mcmc_per_state(
    ...         n_states=3,
    ...         n_dims=[2, 3, 1],
    ...         n_walkers=32,
    ...         n_steps=1000,
    ...         pos=initial_positions,
    ...         log_posterior=my_log_posterior,
    ...         forward_pool=forward_pool
    ...     )
    """

    random.seed(seed)

    # Early validation of forward pool if provided
    if forward_pool is not None:
        from ..utils.forward_context import set_forward_pool, clear_forward_pool
        set_forward_pool(forward_pool)  # Validates map() method
        clear_forward_pool()  # Clear after validation

    if not isinstance(n_walkers, list):
        n_walkers = [n_walkers] * n_states
    if not isinstance(discard, list):
        discard = [discard] * n_states
    if not isinstance(thin, list):
        thin = [thin] * n_states
    if not isinstance(n_steps, list):
        n_steps = [n_steps] * n_states

    if auto_thin:
        # ignore thinning factor because we are post thinning by the auto-correlation times
        # burn_in is also calculated from the auto-correlation time
        thin = [1] * n_states
        discard = [0] * n_states

    if verbose:
        print("\nRunning within-state sampler separately on each state")
        print("\nNumber of walkers               : ", n_walkers)
        print("\nNumber of states being sampled: ", n_states)
        print("Dimensions of each state: ", n_dims)
        if state_pool is not None:
            print("Using state-level parallelism")
        if emcee_pool is not None:
            print("Using walker-level parallelism")

    # Prepare emcee pool configuration to avoid pickling issues
    emcee_pool_config = None
    if emcee_pool is not None:
        # Determine pool type and configuration
        if hasattr(emcee_pool, '__class__'):
            pool_class_name = emcee_pool.__class__.__name__
            if pool_class_name == 'ProcessPoolExecutor':
                emcee_pool_config = {
                    'type': 'ProcessPoolExecutor',
                    'kwargs': {'max_workers': emcee_pool._max_workers}
                }
            elif pool_class_name == 'ThreadPoolExecutor':
                emcee_pool_config = {
                    'type': 'ThreadPoolExecutor',
                    'kwargs': {'max_workers': emcee_pool._max_workers}
                }
    
    # Prepare forward pool configuration to avoid pickling issues
    forward_pool_config = None
    if forward_pool is not None and state_pool is not None:
        # Only create config when using state-level parallelism
        if hasattr(forward_pool, '__class__'):
            pool_class_name = forward_pool.__class__.__name__
            if pool_class_name == 'ProcessPoolExecutor':
                forward_pool_config = {
                    'type': 'ProcessPoolExecutor',
                    'kwargs': {'max_workers': forward_pool._max_workers}
                }
            elif pool_class_name == 'ThreadPoolExecutor':
                forward_pool_config = {
                    'type': 'ThreadPoolExecutor',
                    'kwargs': {'max_workers': forward_pool._max_workers}
                }

    # Prepare state processing arguments
    state_args = []
    for i in range(n_states):
        args_dict = {
            'state_idx': i,
            'log_posterior': log_posterior,
            'n_walkers': n_walkers[i],
            'pos': pos[i],
            'n_steps': n_steps[i],
            'discard': discard[i],
            'thin': thin[i],
            'skip_initial_state_check': skip_initial_state_check,
            'verbose': verbose,
            **kwargs
        }

        # Add pool configs for state-level parallelism
        if state_pool is not None:
            args_dict['emcee_pool_config'] = emcee_pool_config
            if forward_pool_config is not None:
                args_dict['forward_pool_config'] = forward_pool_config
            # Don't pass the actual pool objects
        else:
            # For sequential processing, pass the pools directly
            args_dict['emcee_pool'] = emcee_pool
            args_dict['forward_pool'] = forward_pool

        state_args.append(args_dict)

    # Process states in parallel or sequentially
    if state_pool is not None:
        # Use provided state pool for parallel processing
        results = list(state_pool.map(_process_single_state, state_args))
    elif n_state_processors is not None and n_state_processors > 1:
        # Create internal ProcessPoolExecutor for state parallelism
        with ProcessPoolExecutor(max_workers=n_state_processors) as executor:
            results = list(executor.map(_process_single_state, state_args))
    else:
        # Sequential processing (original behavior)
        # For sequential, we can pass pools directly since no pickling needed
        for args in state_args:
            if 'emcee_pool_config' in args:
                del args['emcee_pool_config']
            if 'forward_pool_config' in args:
                del args['forward_pool_config']
            args['emcee_pool'] = emcee_pool
            args['forward_pool'] = forward_pool
        results = [_process_single_state(args) for args in state_args]

    # Unpack results
    samples = [result[0] for result in results]
    log_posterior_ens = [result[1] for result in results]
    auto_correlation = [result[2] for result in results]

    if auto_thin:
        samples, log_posterior_ens = _perform_auto_thinning(
            samples, log_posterior_ens, auto_correlation, verbose=verbose
        )

    return samples, log_posterior_ens


def process_state(
    log_posterior: Callable[[FloatArray], float],
    n_walkers: int,
    pos: FloatArray,
    n_steps: int,
    discard: int = 0,
    thin: int = 1,
    verbose: bool = True,
    pool: Any | None = None,
    forward_pool: Any | None = None,
    forward_pool_config: dict | None = None,
    **kwargs: Any,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Get the posterior samples, log probabilities, and autocorrelation times for a single state."""
    
    # Always use the picklable wrapper when forward pool is provided
    # This handles all parallelism scenarios cleanly
    if forward_pool is not None or forward_pool_config is not None:
        # Create config from pool if not provided
        if forward_pool_config is None and forward_pool is not None:
            if hasattr(forward_pool, '__class__'):
                pool_class_name = forward_pool.__class__.__name__
                if pool_class_name == 'ProcessPoolExecutor':
                    forward_pool_config = {
                        'type': 'ProcessPoolExecutor',
                        'kwargs': {'max_workers': forward_pool._max_workers}
                    }
                elif pool_class_name == 'ThreadPoolExecutor':
                    forward_pool_config = {
                        'type': 'ThreadPoolExecutor',
                        'kwargs': {'max_workers': forward_pool._max_workers}
                    }
        
        # Use the wrapper for picklable log_posterior
        log_prob_func = LogPosteriorWithForwardPool(log_posterior, forward_pool_config)
    else:
        log_prob_func = log_posterior
    
    try:
        sampler = perform_sampling_with_emcee(
            log_prob_func=log_prob_func,
            n_walkers=n_walkers,
            n_steps=n_steps,
            initial_state=pos,
            pool=pool,
            progress=verbose,
            **kwargs,
        )
        samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
        if samples is None:
            raise ValueError(
                "Sampler did not return a chain. Check the log_prob function and initial state."
            )

        log_posterior_ens = sampler.get_log_prob(discard=discard, thin=thin, flat=True)
        if log_posterior_ens is None:
            raise ValueError(
                "Sampler did not return log probabilities. Check the log_prob function and initial state."
            )

        autocorr_time = sampler.get_autocorr_time(tol=0)
        if autocorr_time is None:
            raise ValueError(
                "Sampler did not return autocorrelation times. Check the log_prob function and initial state."
            )

        return samples, log_posterior_ens, autocorr_time
    finally:
        # Clean up wrapper if created
        if hasattr(log_prob_func, 'cleanup'):
            log_prob_func.cleanup()


def _perform_auto_thinning(
    samples: list[FloatArray],
    log_posterior_ens: list[FloatArray],
    auto_correlation: list[FloatArray],
    verbose: bool = False,
) -> tuple[list[FloatArray], list[FloatArray]]:
    """Perform auto thinning of samples based on autocorrelation times.

    Thin the chains using the maximum auto_correlation function for each state to get independent samples for fitting.
    emcee manual suggests tau = sampler.get_autocorr_time(); burn_in = int(2 * np.max(tau)); thin = int(0.5 * np.min(tau))
    """
    if verbose:
        print("Performing auto thinning of ensemble...")

    samples_auto = []
    log_posterior_ens_auto = []
    for i in range(len(samples)):
        thin = int(np.ceil(0.5 * auto_correlation[i].min()))
        burn_in = int(np.ceil(2.0 * auto_correlation[i].max()))
        samples_auto.append(samples[i][burn_in::thin])
        log_posterior_ens_auto.append(log_posterior_ens[i][burn_in::thin])

    return samples_auto, log_posterior_ens_auto


def _process_single_state(state_args: dict) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Process a single state for parallel state-level execution.
    
    This function is designed to be called by pool.map() for state-level parallelism.
    
    Parameters
    ----------
    state_args : dict
        Dictionary containing all arguments needed to process a single state.
        
    Returns
    -------
    tuple
        Samples, log posterior values, and autocorrelation times for the state.
    """
    state_idx = state_args['state_idx']
    log_posterior = state_args['log_posterior']

    # Handle emcee pool creation (avoid pickling issues)
    emcee_pool_config = state_args.get('emcee_pool_config', None)
    emcee_pool = None

    if emcee_pool_config is not None:
        pool_type = emcee_pool_config['type']
        pool_kwargs = emcee_pool_config.get('kwargs', {})

        if pool_type == 'ProcessPoolExecutor':
            from concurrent.futures import ProcessPoolExecutor
            emcee_pool = ProcessPoolExecutor(**pool_kwargs)
        elif pool_type == 'ThreadPoolExecutor':
            from concurrent.futures import ThreadPoolExecutor
            emcee_pool = ThreadPoolExecutor(**pool_kwargs)
        # Add other pool types as needed
    
    # Get forward pool config and actual pool (if in sequential mode)
    forward_pool_config = state_args.get('forward_pool_config', None)
    forward_pool = state_args.get('forward_pool', None)  # May be passed directly in sequential mode

    # Create partial log posterior for this state
    _log_posterior = partial(log_posterior, state=state_idx)

    # Remove state-specific args from kwargs
    process_kwargs = {k: v for k, v in state_args.items()
                     if k not in ['state_idx', 'log_posterior', 'emcee_pool_config', 'forward_pool_config', 'forward_pool']}
    
    # Pass both forward pool and config to process_state
    process_kwargs['forward_pool'] = forward_pool
    process_kwargs['forward_pool_config'] = forward_pool_config

    try:
        result = process_state(
            log_posterior=_log_posterior,
            pool=emcee_pool,
            **process_kwargs
        )
    finally:
        # Clean up emcee pool if we created it
        if emcee_pool is not None and emcee_pool_config is not None:
            emcee_pool.shutdown(wait=True)

    return result
