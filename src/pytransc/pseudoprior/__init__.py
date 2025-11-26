"""Subpackage for pseudo-prior distributions."""

from .auto_pseudo import PseudoPriorBuilders, build_auto_pseudo_prior


def get_available_pseudo_priors() -> list[str]:
    """Get a list of available pseudo-prior types."""
    return [option.name for option in PseudoPriorBuilders]


__all__ = [
    "build_auto_pseudo_prior",
    "get_available_pseudo_priors",
    "PseudoPriorBuilders",
]
