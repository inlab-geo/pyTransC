"""Forward pool context for log_likelihood function access.

This module provides a mechanism for log_likelihood functions to access forward pools
that were passed to the samplers. It uses global variables that are automatically
process-local in multiprocessing environments.

The typical usage pattern is:
1. User passes forward_pool to a sampler function
2. Sampler sets the pool before calling log_posterior
3. Log_posterior function gets the pool via get_forward_pool()
4. Sampler clears the pool after log_posterior call

Examples
--------
In a log_posterior function:

>>> from pytransc.utils.forward_context import get_forward_pool
>>> def log_posterior(params, state):
...     forward_pool = get_forward_pool()
...     if forward_pool is not None:
...         # Use pool for parallel forward solver calls
...         results = forward_pool.map(forward_solver, param_chunks)
...     else:
...         # Sequential execution
...         results = [forward_solver(p) for p in param_chunks]
"""

# Global variable for forward pool access (process-local in multiprocessing)
_forward_pool = None


def set_forward_pool(pool):
    """Set forward pool in current process.

    Parameters
    ----------
    pool : Any
        Pool object with map() method. Typically ProcessPoolExecutor,
        ThreadPoolExecutor, or schwimmbad pools.

    Raises
    ------
    ValueError
        If pool does not have a map() method.
    RuntimeError
        If setting the pool fails for any reason.

    Examples
    --------
    >>> from concurrent.futures import ProcessPoolExecutor
    >>> with ProcessPoolExecutor(max_workers=4) as pool:
    ...     set_forward_pool(pool)
    """
    global _forward_pool

    # Validate pool has map() method
    if pool is not None and not hasattr(pool, 'map'):
        raise ValueError("Forward pool must have a 'map' method")

    try:
        _forward_pool = pool
    except Exception as e:
        raise RuntimeError(f"Failed to set forward pool: {e}") from e


def get_forward_pool():
    """Get forward pool from current process.

    Returns
    -------
    Pool or None
        The forward pool if available, None otherwise.

    Examples
    --------
    >>> forward_pool = get_forward_pool()
    >>> if forward_pool is not None:
    ...     # Use pool for parallel execution
    ...     results = forward_pool.map(my_function, data_chunks)
    """
    return _forward_pool


def clear_forward_pool():
    """Clear forward pool from current process.

    This should be called after log_posterior evaluation to clean up
    the pool reference and prevent memory leaks.

    Raises
    ------
    RuntimeError
        If clearing the pool fails for any reason.

    Examples
    --------
    >>> clear_forward_pool()  # Pool reference is cleared
    >>> assert get_forward_pool() is None
    """
    global _forward_pool
    try:
        _forward_pool = None
    except Exception as e:
        raise RuntimeError(f"Failed to clear forward pool: {e}") from e
