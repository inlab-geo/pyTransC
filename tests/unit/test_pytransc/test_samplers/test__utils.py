"""Test sampler utilities."""

import numpy as np

from pytransc.samplers._utils import count_visits_to_states


def test_count_visits_to_states():
    """Test the count_visits_to_states function."""

    # Example state chain and number of states
    state_chain = np.array([0, 1, 0, 2, 1, 0])
    n_states = 3

    # Expected output
    expected_counts = np.array(
        [[1, 0, 0], [1, 1, 0], [2, 1, 0], [2, 1, 1], [2, 2, 1], [3, 2, 1]]
    )

    # Call the function and check the result
    counts = count_visits_to_states(state_chain, n_states)
    assert np.array_equal(counts, expected_counts)


def test_count_visits_to_states_empty():
    """Test the count_visits_to_states function with an empty state chain.

    This is the case e.g. at initialisation of the sampler.
    """

    # Example empty state chain and number of states
    state_chain = np.array([], dtype=int)
    n_states = 3

    # Expected output is an empty array
    expected_counts = np.empty((0, n_states), dtype=int)

    # Call the function and check the result
    counts = count_visits_to_states(state_chain, n_states)
    assert np.array_equal(counts, expected_counts)
