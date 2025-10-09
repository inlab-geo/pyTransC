"""Tests for custom pseudo-prior."""

import numpy as np

from pytransc.utils.types import FloatArray


def log_pseudo_prior_fn(x: FloatArray, state: int) -> float:
    """Example log pseudo-prior function."""
    return state * x.sum()


def draw_deviate_fn(state: int) -> FloatArray:
    """Example draw deviate function."""
    return np.ones(state)


def test_custom_pseudo_prior():
    """Test custom pseudo-prior."""
    from pytransc.pseudoprior.custom import build_custom_pseudo_prior

    custom_pseudo = build_custom_pseudo_prior(
        ensemble_per_state=[],  # not used in custom builder
        log_pseudo_prior_fn=log_pseudo_prior_fn,
        draw_deviate_fn=draw_deviate_fn,
    )

    x = np.array([1.0, 2.0, 3.0])
    state = 2

    log_density = custom_pseudo(x, state)
    deviate = custom_pseudo.draw_deviate(state)

    assert log_density == state * x.sum()
    assert np.array_equal(deviate, np.ones(state))
