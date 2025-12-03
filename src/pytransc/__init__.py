"""PyTransC: A Python library for Trans-conceptual Bayesian sampling.

PyTransC implements various trans-conceptual MCMC algorithms for Bayesian inference
across models of different states. The package provides:

- Product space sampling using emcee
- State-jump sampling with pseudo-priors
- Ensemble resampling methods
- Analysis tools for trans-conceptual chains
- Utility functions for pseudo-prior construction

Examples
--------
Basic usage with product space sampler:

    >>> from pytransc.samplers import run_product_space_sampler
    >>> # Set up your log_posterior and log_pseudo_prior functions
    >>> results = run_product_space_sampler(
    ...     nwalkers=32, nsteps=1000,
    ...     pos_initial=initial_positions,
    ...     pos_state=initial_states,
    ...     log_posterior=my_log_posterior,
    ...     log_pseudo_prior=my_log_pseudo_prior
    ... )
"""

__version__ = "0.3.0a1"
