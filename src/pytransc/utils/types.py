"""Custom types for pytransc."""

from typing import Annotated, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

# See https://medium.com/data-science-collective/do-more-with-numpy-array-type-hints-annotate-validate-shape-dtype-09f81c496746
# for guidance on numpy type annotations.
# These types are not actually supported by type checkers, so this is more for documentation purposes.
# Current numpy type annotations only specify the dtype, not the shape.
IntArray: TypeAlias = npt.NDArray[np.integer]
FloatArray: TypeAlias = npt.NDArray[np.floating]
MultiWalkerStateChain: TypeAlias = Annotated[IntArray, "(n_walkers, n_steps)"]
MultiWalkerModelChain: TypeAlias = Annotated[
    list[list[FloatArray]],
    "(n_walkers, n_steps, n_dims[state_i])",
]
StateOrderedEnsemble: TypeAlias = list[FloatArray]


class MultiStateDensity(Protocol):
    """Protocol for multi-state density functions.

    This protocol defines the interface for functions that can evaluate
    log-density values at points in different states. These functions
    are not necessarily normalized.

    Used by all trans-conceptual samplers for posterior evaluation.
    """

    def __call__(self, x: FloatArray, state: int) -> float:
        """Evaluate the log-density at point x in the given state.

        Parameters
        ----------
        x : FloatArray
            Input point where the density is evaluated. Shape should match
            the parameter space dimension for the given state.
        state : int
            The state index (0-based) for which the density is evaluated.

        Returns
        -------
        float
            Log-density value at x in the specified state.
        """
        ...


class MultiStateDraw(Protocol):
    """Protocol for multi-state density functions that support drawing."""

    def __call__(self, state: int) -> FloatArray:
        """Draw a random sample from the distribution for the given state.

        Parameters
        ----------
        state : int
            The state index from which to draw the sample.

        Returns
        -------
        FloatArray
            A random sample from the distribution in the specified state.
            Shape should match the parameter space dimension for that state.
        """
        ...


class SampleableMultiStateDensity(MultiStateDensity, MultiStateDraw, Protocol):
    """Protocol for multi-state density functions that support sampling.

    This protocol extends MultiStateDensity to include the ability to
    generate random samples from the distribution. This is primarily
    used by the state-jump sampler for drawing from pseudo-priors
    when proposing between-state moves.
    """


class ProposableMultiStateDensity(MultiStateDensity, Protocol):
    """Protocol for multi-state density functions that support proposals.

    This protocol extends MultiStateDensity to include the ability to
    generate proposal moves within a state. Used by the state-jump
    sampler for within-state moves.
    """

    def propose(self, x: FloatArray, state: int) -> FloatArray:
        """Propose a new point based on the current point x.

        Parameters
        ----------
        x : FloatArray
            Current point in the parameter space for the given state.
        state : int
            The state index for which the proposal is generated.

        Returns
        -------
        FloatArray
            Proposed new point in the same state. Shape should match
            the input parameter x.
        """
        ...


class MultiStateMultiWalkerResult(Protocol):
    """Protocol for results from multi-state multi-walker samplers.

    This protocol defines the interface for objects that store the results
    of trans-conceptual MCMC sampling with multiple walkers. It provides
    access to state chains, model chains, and visit statistics.
    """

    @property
    def state_chain(self) -> MultiWalkerStateChain:
        """State visitation sequence for each walker.

        Returns
        -------
        MultiWalkerStateChain
            Array of shape (n_walkers, n_steps) containing the state
            index visited by each walker at each step.

        Expected shape is (n_walkers, n_steps).
        """
        ...

    @property
    def model_chain(self) -> MultiWalkerModelChain:
        """Model chain for each walker.

        Expected shape is (n_walkers, n_steps, n_dims).
        """
        ...

    @property
    def state_chain_tot(self) -> IntArray:
        """Running totals of states visited along the Markov chains."""
        ...

    @property
    def n_walkers(self) -> int:
        """Number of walkers in the sampler."""
        ...

    @property
    def n_steps(self) -> int:
        """Number of steps in the sampler."""
        ...

    @property
    def n_states(self) -> int:
        """Number of states in the sampler."""
        ...
