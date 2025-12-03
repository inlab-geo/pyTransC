"""Test the product space sampler."""

import numpy as np
import pytest
from numpy.random import Generator, default_rng

from pytransc.samplers.product_space import (
    ProductSpace,
    _get_initial_product_space_positions,
    product_space_log_prob,
)


@pytest.fixture
def rng() -> Generator:
    """Fixture to provide a random number generator."""
    # Use a fixed seed for reproducibility in tests
    return default_rng(42)


@pytest.fixture
def product_space() -> ProductSpace:
    """Fixture to create a ProductSpace instance for testing."""

    return ProductSpace(n_dims=[3, 4, 5])


def test_product_space_n_states(product_space: ProductSpace) -> None:
    """Test the number of states in the product space."""
    assert product_space.n_states == 3


def test_product_space_total_n_dim(product_space: ProductSpace) -> None:
    """Test the total number of dimensions in the product space."""
    assert product_space.total_n_dim == 13  # 3 + 4 + 5 + 1 (for state index)


def test_model_vectors2product_space(
    product_space: ProductSpace, rng: Generator
) -> None:
    """Test conversion of model vectors to product space format."""
    model_vectors = [rng.random(3), rng.random(4), rng.random(5)]
    state = int(rng.integers(0, product_space.n_states))
    product_space_vector = product_space.model_vectors2product_space(
        state, model_vectors
    )

    assert product_space_vector.size == product_space.total_n_dim
    assert product_space_vector[0] == state
    assert np.allclose(product_space_vector[1:4], model_vectors[0])
    assert np.allclose(product_space_vector[4:8], model_vectors[1])
    assert np.allclose(product_space_vector[8:], model_vectors[2])


def test_product_space2model_vectors(
    product_space: ProductSpace, rng: Generator
) -> None:
    """Test conversion of product space vectors to model format."""
    product_space_vector = rng.standard_normal(product_space.total_n_dim)
    state = int(rng.integers(0, product_space.n_states))
    product_space_vector[0] = state  # Set the state index

    _state, model_vectors = product_space.product_space2model_vectors(
        product_space_vector
    )

    assert _state == state
    assert len(model_vectors) == 3
    assert np.allclose(model_vectors[0], product_space_vector[1:4])
    assert np.allclose(model_vectors[1], product_space_vector[4:8])
    assert np.allclose(model_vectors[2], product_space_vector[8:])


def test_model_vectors_round_trip(product_space: ProductSpace, rng: Generator) -> None:
    """Test round-trip conversion of model vectors to product space and back."""
    model_vectors = [rng.random(3), rng.random(4), rng.random(5)]
    state = int(rng.integers(0, product_space.n_states))

    recovered_state, recovered_model_vectors = (
        product_space.product_space2model_vectors(
            product_space.model_vectors2product_space(state, model_vectors)
        )
    )

    assert recovered_state == state
    assert len(recovered_model_vectors) == 3
    assert np.allclose(recovered_model_vectors[0], model_vectors[0])
    assert np.allclose(recovered_model_vectors[1], model_vectors[1])
    assert np.allclose(recovered_model_vectors[2], model_vectors[2])


def test_product_space_round_trip(product_space: ProductSpace, rng: Generator) -> None:
    """Test round-trip conversion of product space vectors."""
    product_space_vector = rng.standard_normal(product_space.total_n_dim)
    state = int(rng.integers(0, product_space.n_states))
    product_space_vector[0] = state  # Set the state index

    recovered_product_space_vector = product_space.model_vectors2product_space(
        *product_space.product_space2model_vectors(product_space_vector)
    )

    assert np.allclose(recovered_product_space_vector, product_space_vector)


def test__clip_state_index(product_space: ProductSpace) -> None:
    """Test the clipping of state indices."""
    assert product_space._clip_state_index(-1) == 0
    assert product_space._clip_state_index(0) == 0
    assert product_space._clip_state_index(1) == 1
    assert product_space._clip_state_index(2) == 2
    assert product_space._clip_state_index(3) == 2
    assert product_space._clip_state_index(4) == 2
    assert product_space._clip_state_index(1.5) == 2
    assert product_space._clip_state_index(2.5) == 2
    assert product_space._clip_state_index(np.pi) == 2
    assert product_space._clip_state_index(-np.pi) == 0


def test_product_space_log_prob(product_space: ProductSpace, rng: Generator) -> None:
    """Test the log probability of a product space vector.

    Uses simple mock log posterior and log pseudo-prior functions for which the expected log probability can be easily calculated.
    """
    model_vectors = [rng.random(3), rng.random(4), rng.random(5)]
    state = int(rng.integers(0, product_space.n_states))
    product_space_vector = product_space.model_vectors2product_space(
        state, model_vectors
    )

    def log_posterior(x: np.ndarray, state: int) -> float:
        """Mock log posterior function for testing."""
        return float(state)

    def log_pseudo_prior(x: np.ndarray, state: int) -> float:
        """Mock log pseudo-prior function for testing."""
        return -float(state)

    (remaining_states := list(range(product_space.n_states))).remove(state)
    expected_log_prob = state - sum(remaining_states)

    log_prob = product_space_log_prob(
        product_space_vector, product_space, log_posterior, log_pseudo_prior
    )

    assert np.isclose(
        log_prob,
        expected_log_prob,
    ), f"Expected log probability {expected_log_prob}, but got {log_prob}"


def test__get_initial_product_space_positions(
    product_space: ProductSpace, rng: Generator
) -> None:
    """Test the initial product space positions generation."""
    n_walkers = 5
    start_states = [
        int(rng.integers(0, product_space.n_states)) for _ in range(n_walkers)
    ]
    start_positions = [rng.random((n_walkers, dim)) for dim in product_space.n_dims]

    pos_ps = _get_initial_product_space_positions(
        n_walkers, start_states, start_positions, product_space
    )

    for walker in range(n_walkers):
        expected_state = start_states[walker]
        expected_model_vectors = [
            start_positions[state][walker] for state in range(product_space.n_states)
        ]
        pos = pos_ps[walker]

        recovered_state, recovered_model_vectors = (
            product_space.product_space2model_vectors(pos)
        )
        assert expected_state == recovered_state
        for expected_mv, recovered_mv in zip(
            expected_model_vectors, recovered_model_vectors
        ):
            assert np.allclose(expected_mv, recovered_mv), (
                f"Expected model vector {expected_mv}, but got {recovered_mv}"
            )
