"""Tests for custom pseudo-prior."""

import numpy as np
import pytest

from pytransc.pseudoprior.custom import CustomPseudoPrior, build_custom_pseudo_prior
from pytransc.utils.types import MultiStateDensity, MultiStateDraw


@pytest.fixture
def custom_pseudo(
    log_pseudo_prior_fn: MultiStateDensity, draw_deviate_fn: MultiStateDraw
) -> CustomPseudoPrior:
    """Fixture providing a custom pseudo-prior."""

    return build_custom_pseudo_prior(
        ensemble_per_state=[],
        log_pseudo_prior_fn=log_pseudo_prior_fn,
        draw_deviate_fn=draw_deviate_fn,
    )


def test_custom_pseudo_prior_valid_initialization(
    custom_pseudo: CustomPseudoPrior,
    log_pseudo_prior_fn: MultiStateDensity,
    draw_deviate_fn: MultiStateDraw,
) -> None:
    """Test custom pseudo-prior initialization."""

    assert isinstance(custom_pseudo, CustomPseudoPrior)
    assert custom_pseudo._log_pseudo_prior_fn is log_pseudo_prior_fn
    assert custom_pseudo._draw_deviate_fn is draw_deviate_fn


def test_custom_pseudo_prior_invalid_initialization(
    log_pseudo_prior_fn: MultiStateDensity, draw_deviate_fn: MultiStateDraw
) -> None:
    """Test custom pseudo-prior invalid initialization."""

    with pytest.raises(TypeError, match="log_pseudo_prior_fn must be callable"):
        build_custom_pseudo_prior(
            ensemble_per_state=[],
            log_pseudo_prior_fn="not_a_function",  # type: ignore
            draw_deviate_fn=draw_deviate_fn,
        )

    with pytest.raises(TypeError, match="draw_deviate_fn must be callable"):
        build_custom_pseudo_prior(
            ensemble_per_state=[],
            log_pseudo_prior_fn=log_pseudo_prior_fn,
            draw_deviate_fn="not_a_function",  # type: ignore
        )

    with pytest.raises(TypeError, match="missing 2 required keyword-only arguments"):
        build_custom_pseudo_prior(ensemble_per_state=[])  # type: ignore


def test_custom_pseudo_prior_log_density(custom_pseudo: CustomPseudoPrior) -> None:
    """Test custom pseudo-prior."""

    x = np.array([1.0, 2.0, 3.0])
    state = 2

    log_density = custom_pseudo(x, state)
    assert log_density == 12.0


def test_custom_pseudo_prior_deviate(custom_pseudo: CustomPseudoPrior) -> None:
    """Test custom pseudo-prior draw deviate."""

    state = 2
    deviate = custom_pseudo.draw_deviate(state)
    assert np.array_equal(deviate, np.ones(state))
