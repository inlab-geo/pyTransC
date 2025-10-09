"""Module for custom pseudo-prior."""

from ..utils.types import (
    FloatArray,
    MultiStateDensity,
    MultiStateDraw,
)


class CustomPseudoPrior:
    """Class for custom pseudo-prior."""

    def __init__(
        self, log_pseudo_prior_fn: MultiStateDensity, draw_deviate_fn: MultiStateDraw
    ) -> None:
        """
        Initialize the custom pseudo-prior.

        Parameters
        ----------
        log_pseudo_prior_fn : MultiStateDensity
            Function to evaluate the log pseudo-prior density.
        draw_deviate_fn : MultiStateDraw
            Function to draw a random deviate from the pseudo-prior.
        """
        self._log_pseudo_prior_fn = log_pseudo_prior_fn
        self._draw_deviate_fn = draw_deviate_fn

    def __call__(self, x: FloatArray, state: int) -> float:
        """Evaluate the log pseudo-prior density."""
        return self._log_pseudo_prior_fn(x, state)

    def draw_deviate(self, state: int) -> FloatArray:
        """Draw a random deviate from the pseudo-prior for a given state."""
        return self._draw_deviate_fn(state)


def build_custom_pseudo_prior(
    ensemble_per_state: list[
        FloatArray
    ],  # this is here just to match the Protocol signature
    *,
    log_pseudo_prior_fn: MultiStateDensity,
    draw_deviate_fn: MultiStateDraw,
) -> CustomPseudoPrior:
    """
    Build a custom pseudo-prior function.

    Args:
        log_pseudo_prior_fn (MultiStateDensity): Function to evaluate the log pseudo-prior density.
        draw_deviate_fn (MultiStateDraw): Function to draw a random deviate from the pseudo-prior.

    Returns:
        CustomPseudoPrior: Instance of CustomPseudoPrior.
    """
    return CustomPseudoPrior(log_pseudo_prior_fn, draw_deviate_fn)
