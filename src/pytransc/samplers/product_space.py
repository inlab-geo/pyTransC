"""Product-Space Sampling for TransC."""

import random
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from emcee import EnsembleSampler

from ..utils.types import (
    FloatArray,
    IntArray,
    MultiStateDensity,
    MultiWalkerModelChain,
    MultiWalkerStateChain,
)
from ._emcee import perform_sampling_with_emcee


@dataclass
class ProductSpace:
    """Define the product space for TransC sampling."""

    n_dims: list[int]

    @property
    def n_states(self) -> int:
        """Number of states in the product space."""
        return len(self.n_dims)

    @property
    def total_n_dim(self) -> int:
        """Total number of dimensions in the product space.

        Sum of dimensions of all states plus one for the state index.
        """
        return sum(self.n_dims) + 1

    def model_vectors2product_space(
        self,
        state: int,
        model_vectors: list[FloatArray],
    ) -> FloatArray:
        """Convert a list of model vectors to product space format with the selected state."""
        if not (0 <= state < self.n_states):
            raise ValueError(
                f"State index {state} is out of bounds for product space with {self.n_states} states."
            )
        if any(
            mv.size != expected_nd
            for mv, expected_nd in zip(model_vectors, self.n_dims)
        ):
            raise ValueError(
                "Model vectors do not match the dimensions of the product space."
            )
        return np.concatenate([[state], *model_vectors])

    def product_space2model_vectors(
        self,
        product_space_vector: FloatArray,
    ) -> tuple[int, list[FloatArray]]:
        """Convert a product space vector back to a list of model vectors."""
        if product_space_vector.size != self.total_n_dim:
            raise ValueError(
                f"Product space vector size {product_space_vector.size} does not match total dimensions {self.total_n_dim}."
            )
        state = self._clip_state_index(product_space_vector[0])

        k = 1
        model_vectors = []
        for n_dim in self.n_dims:
            model_vectors.append(product_space_vector[k : k + n_dim])
            k += n_dim
        return state, model_vectors

    def _clip_state_index(self, state: float) -> int:
        """Clip the state index to ensure it is within valid bounds.

        The product space vector will in principle always have dtype=float, so we need to ensure the state index is an integer.
        """
        # np.rint rounds to the nearest integer. In the unlikely case of a tie, it rounds to the nearest even integer e.g. 2.5 -> 2.0 (round down) 3.5 -> 4.0 (round up)
        # clip to valid state index range
        return int(np.clip(np.rint(state), 0, self.n_states - 1))


def _get_states_from_emcee_sampler(
    product_space_sampler: EnsembleSampler,
) -> MultiWalkerStateChain:
    """Get the states from the emcee sampler."""
    chain = product_space_sampler.get_chain()
    if chain is None or chain.size == 0:
        raise ValueError("The sampler chain is empty or not initialized.")
    state_chain = chain[:, :, 0].T
    return np.rint(state_chain).astype("int")


def _get_models_from_emcee_sampler(
    product_space_sampler: EnsembleSampler,
    n_dims: list[int],
) -> MultiWalkerModelChain:
    """Get the model vectors from the emcee sampler."""
    samples = product_space_sampler.get_chain()
    if samples is None or samples.size == 0:
        raise ValueError("The sampler chain is empty or not initialized.")

    ind1 = np.cumsum(n_dims) + 1
    ind0 = np.append(np.array([1]), ind1)

    state_chain = np.rint(samples[:, :, 0].T).astype("int")
    n_walkers, n_steps = np.shape(state_chain)
    model_chain = []
    for i in range(n_walkers):
        m = []
        for j in range(n_steps):
            m.append(
                samples[
                    j,
                    i,
                    ind0[state_chain[i, j]] : ind1[state_chain[i, j]],
                ]
            )
        model_chain.append(m)

    return model_chain


@dataclass(init=False)
class MultiWalkerProductSpaceChain:
    """Data class to hold the results of the product space sampler.

    This will basically wrap emcee's EnsembleSampler and provide a more convenient interface for accessing the results that is more consistent with the rest of pyTransC.

    This class can only be initialised using the `from_emcee` class method, which will extract the necessary information from an `emcee.EnsembleSampler` instance.
    """

    _n_states: int
    _model_chain: MultiWalkerModelChain
    _state_chain: MultiWalkerStateChain

    @classmethod
    def from_emcee(
        cls,
        emcee_sampler: EnsembleSampler,
        n_dims: list[int],
    ) -> "MultiWalkerProductSpaceChain":
        """Create a ProductSpaceChain from an emcee EnsembleSampler."""
        state_chain = _get_states_from_emcee_sampler(emcee_sampler)
        model_chain = _get_models_from_emcee_sampler(emcee_sampler, n_dims)
        n_states = len(n_dims)
        obj = cls.__new__(cls)  # Bypass __init__, since we use init=False
        obj._n_states = n_states
        obj._model_chain = model_chain
        obj._state_chain = state_chain
        return obj

    def __repr__(self) -> str:
        """String representation of the MultiWalkerProductSpaceChain."""
        return (
            f"MultiWalkerProductSpaceChain(n_walkers={self.n_walkers}, "
            f"n_steps={self.n_steps}, n_states={self.n_states})"
        )

    @property
    def n_walkers(self) -> int:
        """Number of walkers in the chain."""
        if not self.model_chain:
            return 0
        return self.state_chain.shape[0]

    @property
    def n_states(self) -> int:
        """Number of states in the chain."""
        return self._n_states

    @property
    def n_steps(self) -> int:
        """Number of steps in the chain."""
        return self.state_chain.shape[1]

    @property
    def model_chain(self) -> MultiWalkerModelChain:
        """Model chain for each walker.

        Expected shape is (n_walkers, n_steps, n_dims).
        """
        return self._model_chain

    @property
    def state_chain(self) -> MultiWalkerStateChain:
        """State chain for each walker.

        Expected shape is (n_walkers, n_steps, n_dims).
        """
        return np.array(self._state_chain)

    @property
    def state_chain_tot(self) -> IntArray:
        """Calculate cumulative state visit counts.

        Returns
        -------
        IntArray
            Array of shape (n_walkers, n_steps, n_states) where each entry [i, j, k]
                        is the number of times walker j has visited state k up to step i.
        """
        visits = np.zeros((self.n_walkers, self.n_steps, self.n_states), dtype=int)
        for i in range(self.n_states):
            visits[:, :, i] = np.cumsum(self.state_chain == i, axis=1)

        return visits


def run_product_space_sampler(
    product_space: ProductSpace,
    n_walkers: int,
    n_steps: int,
    start_positions: list[FloatArray],
    start_states: list[int],
    log_posterior: MultiStateDensity,
    log_pseudo_prior: MultiStateDensity,
    seed: int | None = 61254557,
    progress: bool = False,
    pool: Any | None = None,
    forward_pool: Any | None = None,
    **kwargs,
) -> MultiWalkerProductSpaceChain:
    """Run MCMC sampler over independent states using emcee in trans-C product space.

    This function implements trans-conceptual MCMC sampling by embedding all states
    in a fixed-dimensional product space. The sampler uses the emcee ensemble sampler
    to explore the combined parameter space of all states.

    Parameters
    ----------
    product_space : ProductSpace
        The product space definition containing state dimensions.
    n_walkers : int
        Number of random walkers used by the product space sampler.
    n_steps : int
        Number of MCMC steps required per walker.
    start_positions : list of FloatArray
        Starting positions for walkers, one array per walker containing the
        initial parameter values for the starting state.
    start_states : list of int
        Starting state indices for each walker.
    log_posterior : MultiStateDensity
        Function to evaluate the log-posterior density at location x in state i.
        Must have signature log_posterior(x, state) -> float.
    log_pseudo_prior : MultiStateDensity
        Function to evaluate the log-pseudo-prior density at location x in state i.
        Must have signature log_pseudo_prior(x, state) -> float.
        Note: Must be normalized over respective state spaces.
    seed : int, optional
        Random number seed for reproducible results. Default is 61254557.
    progress : bool, optional
        Whether to display progress information. Default is False.
    pool : Any | None, optional
        User-provided pool for parallel processing. The pool must implement
        a map() method compatible with the standard library's map() function.
        Default is None.
    forward_pool : Any | None, optional
        User-provided pool for parallelizing forward solver calls within
        log_posterior evaluations. If provided, the pool will be made available
        to log_posterior functions via get_forward_pool() from pytransc.utils.forward_context.
        The pool must implement a map() method compatible with the standard library's 
        map() function. Supports ProcessPoolExecutor, ThreadPoolExecutor, 
        and schwimmbad pools. Default is None.
    **kwargs
        Additional keyword arguments passed to the emcee sampler.

    Returns
    -------
    MultiWalkerProductSpaceChain
        Chain results containing state sequences, model parameters, and diagnostics
        for all walkers.

    Notes
    -----
    The product space approach embeds all possible states in a single fixed-dimensional
    space. This allows the use of efficient ensemble samplers like emcee, but requires
    sampling in a higher-dimensional space than any individual state.

    Examples
    --------
    Basic usage:

    >>> ps = ProductSpace(n_dims=[2, 3, 1])
    >>> results = run_product_space_sampler(
    ...     product_space=ps,
    ...     n_walkers=32,
    ...     n_steps=1000,
    ...     start_positions=[[0.5, 0.5], [1.0, 0.0, -1.0], [2.0]],
    ...     start_states=[0, 1, 2],
    ...     log_posterior=my_log_posterior,
    ...     log_pseudo_prior=my_log_pseudo_prior
    ... )

    Using with schwimmbad pools:

    >>> from schwimmbad import MPIPool
    >>> with MPIPool() as pool:
    ...     results = run_product_space_sampler(
    ...         product_space=ps,
    ...         n_walkers=32,
    ...         n_steps=1000,
    ...         start_positions=start_pos,
    ...         start_states=start_states,
    ...         log_posterior=my_log_posterior,
    ...         log_pseudo_prior=my_log_pseudo_prior,
    ...         pool=pool
    ...     )

    Using with forward pool for parallel forward solver calls:

    >>> from concurrent.futures import ProcessPoolExecutor
    >>> with ProcessPoolExecutor(max_workers=4) as forward_pool:
    ...     results = run_product_space_sampler(
    ...         product_space=ps,
    ...         n_walkers=32,
    ...         n_steps=1000,
    ...         start_positions=start_pos,
    ...         start_states=start_states,
    ...         log_posterior=my_log_posterior,
    ...         log_pseudo_prior=my_log_pseudo_prior,
    ...         forward_pool=forward_pool
    ...     )
    """

    random.seed(seed)

    # Early validation of forward pool if provided
    if forward_pool is not None:
        from ..utils.forward_context import set_forward_pool, clear_forward_pool
        set_forward_pool(forward_pool)  # Validates map() method
        clear_forward_pool()  # Clear after validation

    if progress:
        print("\nRunning product space trans-C sampler")
        print("\nNumber of walkers               : ", n_walkers)
        print("Number of states being sampled  : ", product_space.n_states)
        print("Dimensions of each state        : ", product_space.n_dims)

    pos_ps = _get_initial_product_space_positions(
        n_walkers, start_states, start_positions, product_space
    )

    log_func = partial(
        product_space_log_prob,
        product_space=product_space,
        log_posterior=log_posterior,
        log_pseudo_prior=log_pseudo_prior,
        forward_pool=forward_pool,
    )

    sampler = perform_sampling_with_emcee(
        log_prob_func=log_func,
        n_walkers=n_walkers,
        n_steps=n_steps,
        initial_state=pos_ps,
        pool=pool,
        progress=progress,
        **kwargs,
    )

    return MultiWalkerProductSpaceChain.from_emcee(sampler, product_space.n_dims)


def product_space_log_prob(
    x: FloatArray,
    product_space: ProductSpace,
    log_posterior: MultiStateDensity,
    log_pseudo_prior: MultiStateDensity,
    forward_pool=None,
):
    """
    Calculate the combined target density for product space vector i.e. sum of log posterior + log pseudo prior density of all other states.

    here input vector is in product space format.

    Inputs:
    x - float array or list : trans-C vectors in product space format. (length = n_walkers*(1 + sum n_dim[i], i=1,...,n_states))
    log_posterior()              : user supplied function to evaluate the log-posterior density for the ith state at location x.
                                    calling sequence log_posterior(x,i)
    log_pseudo_prior()           : user supplied function to evaluate the log-pseudo-prior density for the ith state at location x.
                                    calling sequence log_posterior(x,i).
                                    NB: must be normalized over respective state spaces.


    Returns:
    x - float array or list : trans-C vectors in product space format. (length sum ndim[i], i=1,...,n_states)

    """
    if x[0] < -0.5 or x[0] >= product_space.n_states - 0.5:
        return -np.inf

    # Import within function to avoid circular imports
    from ..utils.forward_context import set_forward_pool, clear_forward_pool

    state, m = product_space.product_space2model_vectors(x)
    
    try:
        # Set forward pool before log_posterior call
        if forward_pool is not None:
            set_forward_pool(forward_pool)
        
        log_prob = log_posterior(m[state], state)
        
    finally:
        # Always clean up after call
        if forward_pool is not None:
            clear_forward_pool()
    
    for i in range(product_space.n_states):
        if i != state:
            new = log_pseudo_prior(m[i], i)
            log_prob += new
    return log_prob


def _get_initial_product_space_positions(
    n_walkers: int,
    start_states: list[int],
    start_positions: list[FloatArray],
    product_space: ProductSpace,
) -> FloatArray:
    """Get start positions and states into product space format.

    Args:
        n_walkers (int): Number of walkers.
        start_states (list[int]): List of starting states for each walker.
        start_positions (list[FloatArray]): List of starting positions for each walker.  Each list element is an array of model vectors for each state.  The indexing is start_positions[state][walker], so start_positions[0].shape = (n_walkers, n_dims[0]), start_positions[1].shape = (n_walkers, n_dims[1]), etc.
        product_space (ProductSpace): The product space object.

    Returns:
        FloatArray: The initial positions in product space format.
    """
    pos_ps = np.zeros((n_walkers, product_space.total_n_dim))
    for walker, state in enumerate(start_states):
        model_vectors = [start[walker] for start in start_positions]
        pos_ps[walker] = product_space.model_vectors2product_space(state, model_vectors)
    return pos_ps
