# pytransc

![Python3](https://img.shields.io/badge/python-3.x-brightgreen.svg)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

_Python library for implementing TransC MCMC sampling_


This repository contains source code to implement three Trans-Conceptual MCMC sampling algorithms as described in the article 
[Sambridge, Valentine and Hauser (2025)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024JB030470).


## Installation

```
pip install git+https://github.com/inlab-geo/pytransc
```
## Documentation

This package of with a single class `TransC_Sampler` implementing three separate MCMC samplers across independent model states implemented as functions of the class

`run_product_space_sampler()` - implements a fixed dimension MCMC sampler over the product space of the states, and extracts a TransC/TransD ensemble. 

`run_state_jump_sampler()` - implements an RJ-MCMC style algorithm using pseudo-prior proposals and balance conditions. 

`run_ensemble_resampler()` - implements a single parameter Metropolis sampler over the state indicator variable. Requires posterior ensembles in each state to be precomputed.

Other utility functions include:

`run_mcmc_per_state()` - performs MCMC sampling within each state.

`build_auto_pseudo_prior()` - fits a mixture model to posterior ensembles in each state to act as a pseudo-prior function.

`get_transc_samples()` - creates posterior TransC/TransD ensemble from results of any sample.

Here is the docstring of the function `run_state_jump_sampler()`:

    """Run MCMC sampler over independent states using pre-computed ensembles.

    This function performs trans-conceptual MCMC by resampling from pre-computed
    posterior ensembles in each state. It calculates relative evidence of each state
    by sampling over the ensemble members according to their posterior and pseudo-prior
    densities.

    Parameters
    ----------
    n_walkers : int
        Number of random walkers used by the ensemble resampler.
    n_steps : int
        Number of Markov chain steps to perform per walker.
    n_states : int
        Number of independent states in the problem.
    n_dims : list of int
        List of parameter dimensions for each state.
    log_posterior_ens : StateOrderedEnsemble
        Log-posterior values of ensemble members in each state.
        Format: list of arrays, where each array contains log-posterior values
        for the ensemble members in that state.
    log_pseudo_prior_ens : StateOrderedEnsemble
        Log-pseudo-prior values of ensemble members in each state.
        Format: list of arrays, where each array contains log-pseudo-prior values
        for the ensemble members in that state.
    seed : int, optional
        Random number seed for reproducible results. Default is 61254557.
    state_proposal_weights : list of list of float, optional
        Weights for proposing transitions between states. Should be a matrix
        where element [i][j] is the weight for proposing state j from state i.
        Diagonal elements are ignored. If None, uniform weights are used.
    progress : bool, optional
        Whether to display progress information. Default is False.
    walker_pool : Any | None, optional
        User-provided pool for parallelizing walker execution. The pool must
        implement a map() method compatible with the standard library's map()
        function. Default is None.
    state_pool : Any | None, optional
        User-provided pool for parallelizing state-level operations such as
        pseudo-prior evaluation across states. Currently reserved for future
        enhancements. Default is None.
    forward_pool : Any | None, optional
        User-provided pool for parallelizing forward solver calls within
        log_posterior evaluations. If provided, the pool will be made available
        to log_posterior functions via get_forward_pool() from pytransc.utils.forward_context.
        The pool must implement a map() method compatible with the standard library's 
        map() function. Supports ProcessPoolExecutor, ThreadPoolExecutor, 
        and schwimmbad pools. Default is None.

    Returns
    -------
    MultiWalkerEnsembleResamplerChain
        Chain results containing state sequences, ensemble member indices,
        and diagnostics for all walkers.

    Notes
    -----
    This method requires pre-computed posterior ensembles and their corresponding
    log-density values. The ensembles can be generated using `run_mcmc_per_state()`
    and the pseudo-prior values using automatic fitting routines.

    The algorithm works by:
    1. Selecting ensemble members within states based on posterior weights
    2. Proposing transitions between states based on relative evidence
    3. Accepting/rejecting proposals using Metropolis-Hastings criterion

    Examples
    --------
    >>> results = run_ensemble_resampler(
    ...     n_walkers=32,
    ...     n_steps=1000,
    ...     n_states=3,
    ...     n_dims=[2, 3, 1],
    ...     log_posterior_ens=posterior_ensembles,
    ...     log_pseudo_prior_ens=pseudo_prior_ensembles
    ... )

    Using with forward pool for parallel forward solver calls:

    >>> from concurrent.futures import ProcessPoolExecutor
    >>> with ProcessPoolExecutor(max_workers=4) as forward_pool:
    ...     results = run_ensemble_resampler(
    ...         n_walkers=32,
    ...         n_steps=1000,
    ...         n_states=3,
    ...         n_dims=[2, 3, 1],
    ...         log_posterior_ens=posterior_ensembles,
    ...         log_pseudo_prior_ens=pseudo_prior_ensembles,
    ...         forward_pool=forward_pool
    ...     )
    """

## Example

```python
import numpy as np
from pytransc.samplers import run_state_jump_sampler
```
Detailed examples of showing implementation of all three samplers can be found in

[`examples/Regression`](./examples/Regression/) - Sampling across 4 states in polynomial regression with all three samplers.

[`examples/AirborneEM`](./examples/AirborneEM) - Ensemble Sampler applied to Airborne EM data.

[`examples/Tomography`](./examples/Tomography) - Ensemble Sampler applied to 2D borehole tomography to demonstrate three level parallelism.

## Licensing
`pytransc` is released as BSD-2-Clause licence.

## Citations and Acknowledgments

> *Sambridge, M., Valentine, A. & Hauser, J., 2025. Trans-Conceptual Sampling: Bayesian Inference With Competing Assumptions, JGR Solid Earth, Volume 130, Issue 8, 17 August 2025, e2024JB030470.*








