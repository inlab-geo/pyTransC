"""Testing analysis functions that calculate visits to states in a multi-state, multi-walker simulation."""

from dataclasses import dataclass

import numpy as np
import pytest

from pytransc.analysis.visits import (
    count_state_changes,
    count_total_state_changes,
    get_acceptance_rate_between_states,
    get_relative_marginal_likelihoods,
    walker_average_functions,
)


def test_walker_average_functions():
    """Test the walker average functions."""

    # Simulate final counts of visits of multiple walkers to different states
    # Expected shape: (n_walkers, n_states)
    state_chain = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    # i.e. we have 3 walkers and 4 states
    # walker 0 visited state 0 once, state 1 twice, state 2 three times, and state 3 four times

    # The walker_average_functions average the counts across walkers for each state
    # Expected shape: (n_states,)

    # Test mean function
    mean_result = walker_average_functions["mean"](state_chain)
    assert np.array_equal(mean_result, np.array([5.0, 6.0, 7.0, 8.0]))

    # Test median function
    # in this case, the median is the same as the mean since we have an even distribution
    median_result = walker_average_functions["median"](state_chain)
    assert np.array_equal(median_result, np.array([5.0, 6.0, 7.0, 8.0]))


def test_get_relative_marginal_likelihoods():
    """Test the get_relative_marginal_likelihoods function."""

    expected = np.array([0.1, 0.2, 0.7])

    # Simulate visits in these proportions
    visits_to_states = np.array(
        [expected * 10, expected * 10, expected * 10], dtype=int
    )

    rml = get_relative_marginal_likelihoods(
        visits_to_states,
        # default walker average
    )
    assert np.array_equal(rml, expected)


def test_get_relative_marginal_likelihoods_invalid_average():
    """Test that an error is raised for an invalid walker average function."""

    visits_to_states = np.array([[10, 20, 30], [5, 10, 15]], dtype=int)

    with pytest.raises(KeyError):
        get_relative_marginal_likelihoods(
            visits_to_states,
            walker_average="invalid_average",
        )


def test_count_state_changes():
    """Test the count_total_state_changes function."""

    # Simulate a state chain with some state changes
    state_chain = np.array([[0, 1, 1, 2], [0, 0, 1, 1], [2, 2, 2, 3]])

    # Count the state changes
    changes = count_state_changes(state_chain)

    assert changes.shape == (3,)  # 3 walkers
    assert np.issubdtype(changes.dtype, np.integer)
    assert np.array_equal(changes, np.array([2, 1, 1]))


def test_count_state_changes_with_discard_and_thin():
    """Test the count_state_changes function with discard and thin parameters."""

    # Simulate a state chain with some state changes
    state_chain = np.array([[0, 1, 1, 2], [0, 0, 1, 1], [2, 2, 2, 3]])
    discard = 1
    thin = 2
    # After discard and thinning we have:
    # Walker 0: [1, 2] (1 change)
    # Walker 1: [0, 1] (1 change)
    # Walker 2: [2, 3] (1 change)

    # Count the state changes with discard and thin
    changes = count_state_changes(state_chain, discard=discard, thin=thin)

    assert changes.shape == (3,)  # 3 walkers
    assert np.issubdtype(changes.dtype, np.integer)
    assert np.array_equal(changes, np.array([1, 1, 1]))


def test_count_total_state_changes():
    """Test the count_total_state_changes function."""

    # Simulate a state chain with some state changes
    state_chain = np.array([[0, 1, 1, 2], [0, 0, 1, 1], [2, 2, 2, 3]])

    # Count the total state changes
    total_changes = count_total_state_changes(state_chain)

    assert isinstance(total_changes, int)
    assert total_changes == 4  # Total changes across all walkers


@dataclass
class MultiStateMultiWalkerResult:
    """Dummy class to satisfy the MultiStateMultiWalkerResult protocol."""

    state_chain: np.ndarray
    state_chain_tot: np.ndarray
    model_chain: list
    n_walkers: int
    n_states: int
    n_steps: int


@pytest.fixture
def result():
    """Fixture to provide something that satisfies the MultiStateMultiWalkerResult interface."""
    return MultiStateMultiWalkerResult(
        state_chain=np.array([[0, 1, 1, 2], [0, 0, 1, 1], [2, 2, 2, 3]]),
        state_chain_tot=np.array([1, 2, 3, 4]),  # just a placeholder
        model_chain=[[np.arange(4)] * 4] * 3,  # not important for visits to states
        n_walkers=3,
        n_states=4,
        n_steps=4,
    )


def test_get_acceptance_rate_between_states(result):
    """Test the get_acceptance_rate_between_states function."""

    # Calculate the acceptance rate
    acceptance_rate = get_acceptance_rate_between_states(result)

    assert np.isclose(
        acceptance_rate, 100 * (4 / 12)
    )  # 4 changes out of 12 total steps
