"""Testing the user-facing auto build function for pseudo-priors."""


import pytest

from pytransc.pseudoprior import build_auto_pseudo_prior
from pytransc.utils.exceptions import InputError
from pytransc.utils.types import FloatArray


def test_auto_pseudo_prior_invalid_method(ensemble_per_state: list[FloatArray]) -> None:
    """Test that invalid method raises ValueError."""
    with pytest.raises(InputError, match="Invalid pseudo_prior_type: invalid_method"):
        build_auto_pseudo_prior(ensemble_per_state, pseudo_prior_type="invalid_method")


def test_passing_pseudo_type_as_string(ensemble_per_state: list[FloatArray]) -> None:
    """Test that passing pseudo_prior_type as string works."""
    pseudo = build_auto_pseudo_prior(
        pseudo_prior_type="mean_covariance", ensemble_per_state=ensemble_per_state
    )
    assert pseudo.__class__.__name__ == "MeanCovariancePseudoPrior"
