"""Testing the user-facing auto build function for pseudo-priors."""

from unittest.mock import patch

import numpy as np
import pytest

from pytransc.pseudoprior import build_auto_pseudo_prior
from pytransc.pseudoprior.auto_pseudo import PseudoPriorOptions
from pytransc.utils.exceptions import InputError
from pytransc.utils.types import FloatArray


def test_auto_pseudo_prior_invalid_method() -> None:
    """Test that invalid method raises ValueError."""
    with pytest.raises(InputError, match="Invalid pseudo_prior_type"):
        build_auto_pseudo_prior("invalid_method")


def test_passing_pseudo_type_as_string(ensemble_per_state: list[FloatArray]) -> None:
    """Test that passing pseudo_prior_type as string works."""
    pseudo = build_auto_pseudo_prior(
        pseudo_prior_type="mean_covariance", ensemble_per_state=ensemble_per_state
    )
    assert pseudo.__class__.__name__ == "MeanCovariancePseudoPrior"


def test_custom_missing_functions() -> None:
    """Test error raised when insufficient functions provided for custom method."""

    with pytest.raises(
        InputError,
        match="both log_pseudo_prior_fn and draw_deviate_fn must be provided",
    ):
        build_auto_pseudo_prior(
            pseudo_prior_type=PseudoPriorOptions.CUSTOM,
            log_pseudo_prior_fn=None,
            draw_deviate_fn=None,
        )


def test_no_ensemble_no_log_posterior() -> None:
    """Test error raised when no ensemble and no log posterior provided."""

    with pytest.raises(
        InputError,
        match="log_posterior must be provided if ensemble_per_state is not supplied",
    ):
        build_auto_pseudo_prior(
            pseudo_prior_type=PseudoPriorOptions.MEAN_COVARIANCE,
            ensemble_per_state=None,
            log_posterior_fn=None,
        )


def test_no_ensemble_incomplete_sampling_args() -> None:
    """Test error raised when no ensemble and incomplete sampling args provided."""

    with pytest.raises(
        InputError,
        match="sampling_args must contain 'n_states', 'n_dims', 'n_walkers', 'n_steps', and 'pos' keys",
    ):
        build_auto_pseudo_prior(
            pseudo_prior_type=PseudoPriorOptions.MEAN_COVARIANCE,
            ensemble_per_state=None,
            log_posterior_fn=lambda x, state: -0.5 * (x**2).sum(),
            sampling_args={"n_states": 2, "n_dims": 1},  # incomplete
        )


def test_ensembles_generated() -> None:
    """Test that ensembles are generated when not provided."""

    rng = np.random.default_rng(42)

    with patch(
        "pytransc.pseudoprior.auto_pseudo._generate_missing_ensembles"
    ) as mock_generate:
        mock_generate.return_value = [
            rng.normal(size=(100, 1)),
            rng.normal(size=(50, 3)),
        ]

        def _log_posterior_fn(x: np.ndarray, state: int) -> float:
            return -0.5 * (x**2).sum()

        sampling_args = {
            "n_states": 2,
            "n_dims": 1,
            "n_walkers": 10,
            "n_steps": 50,
            "pos": [[1.0], [2.0]],
            "random_state": 42,
        }

        _ = build_auto_pseudo_prior(
            pseudo_prior_type=PseudoPriorOptions.MEAN_COVARIANCE,
            ensemble_per_state=None,
            log_posterior_fn=_log_posterior_fn,
            sampling_args=sampling_args,
        )

        mock_generate.assert_called_once_with(
            log_posterior=_log_posterior_fn, sampling_args=sampling_args
        )
