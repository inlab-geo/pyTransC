"""Tests for the state jump sampler."""

from copy import deepcopy

import numpy as np
import pytest

from pytransc.samplers.state_jump import (
    MultiWalkerStateJumpChain,
    ProposalType,
    Sample,
    StateJumpChain,
    update_chain,
)


@pytest.fixture
def chain() -> StateJumpChain:
    """Fixture to create a Chain instance for testing."""
    rng = np.random.default_rng(42)

    state_chain = [0, 1, 0, 2, 1, 4]  # all proposals are between states
    n_dims = [state + 1 for state in state_chain]
    model_chain = [rng.random(n_dim) for n_dim in n_dims]

    chain = StateJumpChain(n_states=5)
    accept = True  # accept all proposals for testing

    # The initial sample doesn't actually get added to the chain
    # as there is no "proposal", as such.
    # Creating an initial sample such that the first proposal is a "within state" proposal.
    previous_state = 0

    for state, model in zip(state_chain, model_chain):
        sample = Sample(model=model, state=state)
        proposal = (
            ProposalType.WITHIN_STATE
            if state == previous_state
            else ProposalType.BETWEEN_STATE
        )
        update_chain(chain, sample, proposal, accept)
        previous_state = state

    return chain


@pytest.fixture
def multi_walker_chain(chain: StateJumpChain) -> MultiWalkerStateJumpChain:
    """Fixture to create a MultiWalkerStateJumpChain instance for testing."""
    return MultiWalkerStateJumpChain([chain] * 3)


def test_chain_default_initialization() -> None:
    """Test the default initialization of the StateJumpChain class."""

    chain = StateJumpChain(n_states=5)
    assert chain.n_states == 5
    assert chain.model_chain == []
    assert chain.state_chain == []
    assert chain.accept_within == 0
    assert chain.prop_within == 0
    assert chain.accept_between == 0
    assert chain.prop_between == 0


def test_chain_n_steps(chain: StateJumpChain) -> None:
    """Test the number of steps is correctly calculated."""

    assert chain.n_steps == 6


def test_chain_state_chain_tot(chain: StateJumpChain) -> None:
    """Test the state_chain_tot property."""

    expected_totals = [
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [2, 1, 0, 0, 0],
        [2, 1, 1, 0, 0],
        [2, 2, 1, 0, 0],
        [2, 2, 1, 0, 1],
    ]

    assert np.array_equal(chain.state_chain_tot, expected_totals)


def test_update_chain_accept_within(chain: StateJumpChain) -> None:
    """Test the update_chain function for accepting within-state proposals."""

    initial_chain = deepcopy(chain)

    sample = Sample(model=np.array([1, 2]), state=1)
    proposal_type = ProposalType.WITHIN_STATE
    accepted = True

    update_chain(chain, sample, proposal_type, accepted)

    assert chain.state_chain == initial_chain.state_chain + [sample.state]
    for new, old in zip(chain.model_chain, initial_chain.model_chain + [sample.model]):
        assert np.array_equal(new, old)
    assert chain.accept_within == initial_chain.accept_within + 1
    assert chain.prop_within == initial_chain.prop_within + 1
    assert chain.accept_between == initial_chain.accept_between
    assert chain.prop_between == initial_chain.prop_between


def test_update_chain_accept_between(chain: StateJumpChain) -> None:
    """Test the update_chain function for accepting between-state proposals."""

    initial_chain = deepcopy(chain)

    sample = Sample(model=np.array([3, 4]), state=2)
    proposal_type = ProposalType.BETWEEN_STATE
    accepted = True

    update_chain(chain, sample, proposal_type, accepted)

    assert chain.state_chain == initial_chain.state_chain + [sample.state]
    for new, old in zip(chain.model_chain, initial_chain.model_chain + [sample.model]):
        assert np.array_equal(new, old)
    assert chain.accept_between == initial_chain.accept_between + 1
    assert chain.prop_between == initial_chain.prop_between + 1
    assert chain.accept_within == initial_chain.accept_within
    assert chain.prop_within == initial_chain.prop_within


def test_update_chain_reject_within(chain: StateJumpChain) -> None:
    """Test the update_chain function for rejecting proposals."""

    initial_chain = deepcopy(chain)

    sample = Sample(model=np.array([5, 6]), state=3)
    proposal_type = ProposalType.WITHIN_STATE
    accepted = False
    update_chain(chain, sample, proposal_type, accepted)

    # Even though accepted is False, sample is still added to the chain
    # because the update_chain method takes whatever sample is to be added,
    # regardless of acceptance.
    assert chain.state_chain == initial_chain.state_chain + [sample.state]
    for new, old in zip(chain.model_chain, initial_chain.model_chain + [sample.model]):
        assert np.array_equal(new, old)
    assert chain.accept_within == initial_chain.accept_within
    assert chain.prop_within == initial_chain.prop_within + 1
    assert chain.accept_between == initial_chain.accept_between
    assert chain.prop_between == initial_chain.prop_between


def test_update_chain_reject_between(chain: StateJumpChain) -> None:
    """Test the update_chain function for rejecting between-state proposals."""

    initial_chain = deepcopy(chain)

    sample = Sample(model=np.array([7, 8]), state=5)
    proposal_type = ProposalType.BETWEEN_STATE
    accepted = False

    update_chain(chain, sample, proposal_type, accepted)

    # Even though accepted is False, sample is still added to the chain
    # because the update_chain method takes whatever sample is to be added,
    # regardless of acceptance.
    assert chain.state_chain == initial_chain.state_chain + [sample.state]
    for new, old in zip(chain.model_chain, initial_chain.model_chain + [sample.model]):
        assert np.array_equal(new, old)

    assert chain.accept_between == initial_chain.accept_between
    assert chain.prop_between == initial_chain.prop_between + 1
    assert chain.accept_within == initial_chain.accept_within
    assert chain.prop_within == initial_chain.prop_within


def test_multi_walker_chain_initialization(
    multi_walker_chain: MultiWalkerStateJumpChain,
) -> None:
    """Test the initialization of the MultiWalkerStateJumpChain class."""

    assert multi_walker_chain.n_walkers == 3
    assert multi_walker_chain.n_states == 5


def test_multi_walker_chain_invalid_chains() -> None:
    """Test that MultiWalkerStateJumpChain raises an error for invalid chains."""

    with pytest.raises(
        ValueError, match="All chains must have the same number of states."
    ):
        MultiWalkerStateJumpChain(
            [StateJumpChain(n_states=5), StateJumpChain(n_states=6)]
        )

    with pytest.raises(
        TypeError, match="All chains must be instances of StateJumpChain."
    ):
        MultiWalkerStateJumpChain([StateJumpChain(n_states=5), "not_a_chain"])


def test_multi_walker_chain_state_chain_tot(
    multi_walker_chain: MultiWalkerStateJumpChain,
) -> None:
    """Test the state_chain_tot property of MultiWalkerStateJumpChain."""

    # Simulate some state transitions for each walker

    expected_shape = (3, 6, 5)  # (n_walkers, n_steps, n_states)
    assert multi_walker_chain.state_chain_tot.shape == expected_shape

    expected_totals = [
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [2, 1, 0, 0, 0],
            [2, 1, 1, 0, 0],
            [2, 2, 1, 0, 0],
            [2, 2, 1, 0, 1],
        ]
    ] * 3  # Same totals for each walker

    assert np.array_equal(multi_walker_chain.state_chain_tot, expected_totals)


def test_multi_walker_chain_state_chain(
    multi_walker_chain: MultiWalkerStateJumpChain,
) -> None:
    """Test the state_chain property of MultiWalkerStateJumpChain."""

    expected_shape = (3, 6)  # (n_walkers, n_steps)
    assert multi_walker_chain.state_chain.shape == expected_shape
    assert np.array_equal(multi_walker_chain.state_chain, [[0, 1, 0, 2, 1, 4]] * 3)


def test_multi_walker_chain_model_chain(
    multi_walker_chain: MultiWalkerStateJumpChain,
) -> None:
    """Test the model_chain property of MultiWalkerStateJumpChain."""

    assert len(multi_walker_chain.model_chain) == 3
    for chain in multi_walker_chain.chains:
        n_dims = [state + 1 for state in chain.state_chain]
        assert len(chain.model_chain) == 6
        for model, expected_ndim in zip(chain.model_chain, n_dims):
            assert model.shape == (expected_ndim,)


def test_multi_walker_chain_accept_within(
    multi_walker_chain: MultiWalkerStateJumpChain,
) -> None:
    """Test the accept_within property of MultiWalkerStateJumpChain."""
    assert np.array_equal(multi_walker_chain.accept_within, [1] * 3)


def test_multi_walker_chain_prop_within(
    multi_walker_chain: MultiWalkerStateJumpChain,
) -> None:
    """Test the prop_within property of MultiWalkerStateJumpChain."""

    assert np.array_equal(multi_walker_chain.prop_within, [1] * 3)


def test_multi_walker_chain_accept_between(
    multi_walker_chain: MultiWalkerStateJumpChain,
) -> None:
    """Test the accept_between property of MultiWalkerStateJumpChain."""

    assert np.array_equal(multi_walker_chain.accept_between, [5] * 3)


def test_multi_walker_chain_prop_between(
    multi_walker_chain: MultiWalkerStateJumpChain,
) -> None:
    """Test the prop_between property of MultiWalkerStateJumpChain."""

    assert np.array_equal(multi_walker_chain.prop_between, [5] * 3)
