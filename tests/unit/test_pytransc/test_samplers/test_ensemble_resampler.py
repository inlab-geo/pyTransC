"""Test the ensemble resampler."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from pytransc.samplers.ensemble_resampler import (
    EnsembleResamplerChain,
    Sample,
    _log_prob_sample,
    _propose_member_in_state,
    _propose_state,
    run_ensemble_resampler,
    update_chain,
)

# Set multiprocessing start method for testing
try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    # Already set, ignore
    pass


def test_update_chain_accept():
    """Test the update_chain function with an accepted sample."""
    # Create a sample with dummy data
    sample = Sample(member=0, state=1)

    # Create an EnsembleResamplerChain instance
    chain = EnsembleResamplerChain(n_states=3)

    # Update the chain with the sample
    update_chain(chain, sample, proposal_accepted=True)

    assert chain.member_chain == [0]
    assert chain.state_chain == [1]
    assert chain.n_accepted == 1
    assert chain.n_proposed == 1
    assert np.array_equal(chain.state_chain_tot, np.array([[0, 1, 0]]))


def test_update_chain_reject():
    """Test the update_chain function with a rejected sample."""
    # Create a sample with dummy data
    sample = Sample(member=0, state=1)

    # Create an EnsembleResamplerChain instance
    chain = EnsembleResamplerChain(n_states=3)

    # Update the chain with the sample, but reject it
    update_chain(chain, sample, proposal_accepted=False)

    # Note that the member chain and state chain should still be updated.  In an MCMC context, the Sample will be the previous sample, not the proposed one.
    assert chain.member_chain == [0]
    assert chain.state_chain == [1]
    assert chain.n_accepted == 0  # This is the key difference
    assert chain.n_proposed == 1
    assert np.array_equal(chain.state_chain_tot, np.array([[0, 1, 0]]))


def test__log_prob_sample():
    """Test the _log_prob_sample function."""
    # The log probability of a sample is the difference between the log posterior and the log pseudo-prior.

    sample = Sample(member=2, state=1)
    log_posterior_ensemble = np.array(
        [[-1.0, -2.0, -3.0], [-1.5, -2.5, -3.5], [-2.0, -3.0, -4.0]]
    )
    log_pseudo_prior_ensemble = np.array(
        [[-0.5, -1.5, -2.5], [-1.0, -2.0, -3.0], [-1.5, -2.5, -3.5]]
    )

    assert _log_prob_sample(
        sample, log_posterior_ensemble, log_pseudo_prior_ensemble
    ) == -3.5 - (-3.0)


def test__propose_member_in_state() -> None:
    """Test the _propose_member_in_state function.

    Checking that the proposed member index is within the bounds of the state ensemble size.
    """
    n_samples_in_state = 100_000
    for _ in range(1_000):
        assert 0 <= _propose_member_in_state(n_samples_in_state) < n_samples_in_state


def test__propose_state() -> None:
    """Test the _propose_state function.

    Checking that the proposed state index is within the bounds of the number of states and is NEVER the current state.
    """
    n_states = 5
    current_state = 2
    weights = [[1.0] * n_states] * n_states  # Uniform distribution for simplicity
    for _ in range(1_000):
        state = _propose_state(current_state, n_states, weights)
        assert 0 <= state < n_states
        assert state != current_state


class TestEnsembleResamplerParallelism:
    """Test parallelism functionality in ensemble resampler."""

    @pytest.fixture
    def ensemble_data(self):
        """Create test ensemble data."""
        n_states = 2
        n_dims = [2, 3]
        n_samples = [50, 60]
        
        # Create mock ensemble data
        log_posterior_ens = []
        log_pseudo_prior_ens = []
        
        for i in range(n_states):
            # Mock log posterior values
            log_post = np.random.normal(-10, 2, n_samples[i])
            log_posterior_ens.append(log_post)
            
            # Mock log pseudo prior values
            log_pseudo = np.random.normal(-12, 2, n_samples[i])
            log_pseudo_prior_ens.append(log_pseudo)
        
        return n_states, n_dims, log_posterior_ens, log_pseudo_prior_ens

    def test_sequential_ensemble_resampler(self, ensemble_data):
        """Test sequential ensemble resampler execution."""
        n_states, n_dims, log_posterior_ens, log_pseudo_prior_ens = ensemble_data
        
        results = run_ensemble_resampler(
            n_walkers=8,
            n_steps=20,
            n_states=n_states,
            n_dims=n_dims,
            log_posterior_ens=log_posterior_ens,
            log_pseudo_prior_ens=log_pseudo_prior_ens,
            progress=False
        )
        
        # Validate results
        assert results.n_walkers == 8
        assert results.n_states == n_states
        # Note: ensemble resampler does n_steps - 1 internally
        assert results.n_steps == 19
        assert len(results.chains) == 8

    def test_parallel_ensemble_resampler(self, ensemble_data):
        """Test parallel ensemble resampler execution."""
        from concurrent.futures import ProcessPoolExecutor

        n_states, n_dims, log_posterior_ens, log_pseudo_prior_ens = ensemble_data

        with ProcessPoolExecutor(max_workers=2) as walker_pool:
            results = run_ensemble_resampler(
                n_walkers=8,
                n_steps=20,
                n_states=n_states,
                n_dims=n_dims,
                log_posterior_ens=log_posterior_ens,
                log_pseudo_prior_ens=log_pseudo_prior_ens,
                walker_pool=walker_pool,
                progress=False
            )

        # Validate results
        assert results.n_walkers == 8
        assert results.n_states == n_states
        # Note: ensemble resampler does n_steps - 1 internally
        assert results.n_steps == 19
        assert len(results.chains) == 8

    def test_ensemble_resampler_consistency(self, ensemble_data):
        """Test that parallel and sequential give equivalent results structure."""
        from concurrent.futures import ProcessPoolExecutor

        n_states, n_dims, log_posterior_ens, log_pseudo_prior_ens = ensemble_data

        # Use same seed for reproducibility
        seed = 12345

        # Sequential results
        results_seq = run_ensemble_resampler(
            n_walkers=4,
            n_steps=50,
            n_states=n_states,
            n_dims=n_dims,
            log_posterior_ens=log_posterior_ens,
            log_pseudo_prior_ens=log_pseudo_prior_ens,
            seed=seed,
            progress=False
        )

        # Parallel results
        with ProcessPoolExecutor(max_workers=2) as walker_pool:
            results_par = run_ensemble_resampler(
                n_walkers=4,
                n_steps=50,
                n_states=n_states,
                n_dims=n_dims,
                log_posterior_ens=log_posterior_ens,
                log_pseudo_prior_ens=log_pseudo_prior_ens,
                seed=seed,
                walker_pool=walker_pool,
                progress=False
            )
        
        # Results should have same structure
        assert results_seq.n_walkers == results_par.n_walkers
        assert results_seq.n_states == results_par.n_states
        assert results_seq.n_steps == results_par.n_steps
        
        # State visit patterns should be similar (allowing for randomness)
        seq_visits = np.sum(results_seq.state_chain_tot, axis=0)
        par_visits = np.sum(results_par.state_chain_tot, axis=0)
        
        # Check that both visited all states (basic sanity check)
        assert np.all(seq_visits > 0) or np.all(par_visits > 0)  # At least one should visit all
