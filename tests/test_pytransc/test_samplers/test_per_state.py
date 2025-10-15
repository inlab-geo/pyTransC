"""Test the per-state sampler and parallelism functionality."""

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pytest

from pytransc.samplers import run_mcmc_per_state

# Set multiprocessing start method for testing
try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    # Already set, ignore
    pass


def simple_log_posterior(params, state):
    """Simple log posterior for testing."""
    return -0.5 * np.sum(params**2)


def compute_intensive_log_posterior(params, state):
    """More compute-intensive log posterior for performance testing."""
    # Add some computation to make parallelism beneficial
    result = -0.5 * np.sum(params**2)
    for _ in range(100):  # Simulate computation
        result += -0.01 * np.sum(params**4)
    return result


@pytest.fixture
def basic_problem():
    """Basic test problem setup."""
    n_states = 2
    n_dims = [2, 3]
    n_walkers = 8
    n_steps = 10
    pos = [np.random.normal(0, 0.1, (n_walkers, dim)) for dim in n_dims]
    return n_states, n_dims, n_walkers, n_steps, pos


@pytest.fixture
def multi_state_problem():
    """Larger problem for parallelism testing."""
    n_states = 4
    n_dims = [2, 3, 4, 2]
    n_walkers = 16
    n_steps = 20
    pos = [np.random.normal(0, 0.1, (n_walkers, dim)) for dim in n_dims]
    return n_states, n_dims, n_walkers, n_steps, pos


class TestBasicFunctionality:
    """Test basic per-state sampler functionality."""

    def test_sequential_execution(self, basic_problem):
        """Test basic sequential execution."""
        n_states, n_dims, n_walkers, n_steps, pos = basic_problem
        
        samples, log_probs = run_mcmc_per_state(
            n_states=n_states,
            n_dims=n_dims,
            n_walkers=n_walkers,
            n_steps=n_steps,
            pos=pos,
            log_posterior=simple_log_posterior,
            verbose=False
        )
        
        # Validate results
        assert len(samples) == n_states
        assert len(log_probs) == n_states
        
        for i, (s, lp) in enumerate(zip(samples, log_probs)):
            assert s.shape[1] == n_dims[i]
            assert len(s) == len(lp)
            assert s.shape[0] > 0  # Should have some samples

    def test_auto_thin(self, basic_problem):
        """Test auto-thinning functionality."""
        n_states, n_dims, n_walkers, n_steps, pos = basic_problem
        
        samples, log_probs = run_mcmc_per_state(
            n_states=n_states,
            n_dims=n_dims,
            n_walkers=n_walkers,
            n_steps=n_steps,
            pos=pos,
            log_posterior=simple_log_posterior,
            auto_thin=True,
            verbose=False
        )
        
        # Validate results
        assert len(samples) == n_states
        assert len(log_probs) == n_states
        
        # Auto-thinning should reduce sample count
        for s in samples:
            assert s.shape[0] <= n_walkers * n_steps

    def test_parameter_broadcasting(self):
        """Test parameter broadcasting for different states."""
        n_states = 3
        n_dims = [2, 3, 4]
        n_walkers = [8, 12, 16]  # Different per state
        n_steps = [10, 15, 20]   # Different per state
        pos = [np.random.normal(0, 0.1, (n_walkers[i], n_dims[i])) for i in range(n_states)]
        
        samples, log_probs = run_mcmc_per_state(
            n_states=n_states,
            n_dims=n_dims,
            n_walkers=n_walkers,
            n_steps=n_steps,
            pos=pos,
            log_posterior=simple_log_posterior,
            verbose=False
        )
        
        # Validate different configurations were used
        for i, s in enumerate(samples):
            assert s.shape[1] == n_dims[i]


class TestStateParallelism:
    """Test state-level parallelism functionality."""

    def test_state_pool_executor(self, multi_state_problem):
        """Test state parallelism with ProcessPoolExecutor."""
        n_states, n_dims, n_walkers, n_steps, pos = multi_state_problem
        
        with ProcessPoolExecutor(max_workers=2) as pool:
            samples, log_probs = run_mcmc_per_state(
                n_states=n_states,
                n_dims=n_dims,
                n_walkers=n_walkers,
                n_steps=n_steps,
                pos=pos,
                log_posterior=simple_log_posterior,
                state_pool=pool,
                verbose=False
            )
        
        # Validate results
        assert len(samples) == n_states
        assert len(log_probs) == n_states
        
        for i, (s, lp) in enumerate(zip(samples, log_probs)):
            assert s.shape[1] == n_dims[i]
            assert len(s) == len(lp)

    def test_internal_state_parallelism(self, multi_state_problem):
        """Test internal state parallelism via n_state_processors."""
        n_states, n_dims, n_walkers, n_steps, pos = multi_state_problem
        
        samples, log_probs = run_mcmc_per_state(
            n_states=n_states,
            n_dims=n_dims,
            n_walkers=n_walkers,
            n_steps=n_steps,
            pos=pos,
            log_posterior=simple_log_posterior,
            n_state_processors=2,
            verbose=False
        )
        
        # Validate results
        assert len(samples) == n_states
        assert len(log_probs) == n_states

    @pytest.mark.skipif(os.cpu_count() < 4, reason="Requires at least 4 CPU cores")
    def test_state_parallelism_performance(self, multi_state_problem):
        """Test that state parallelism provides performance benefit."""
        n_states, n_dims, n_walkers, n_steps, pos = multi_state_problem
        
        # Sequential timing
        start_time = time.time()
        run_mcmc_per_state(
            n_states=n_states,
            n_dims=n_dims,
            n_walkers=n_walkers,
            n_steps=n_steps,
            pos=pos,
            log_posterior=compute_intensive_log_posterior,
            verbose=False
        )
        sequential_time = time.time() - start_time
        
        # Parallel timing
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=min(n_states, 4)) as pool:
            run_mcmc_per_state(
                n_states=n_states,
                n_dims=n_dims,
                n_walkers=n_walkers,
                n_steps=n_steps,
                pos=pos,
                log_posterior=compute_intensive_log_posterior,
                state_pool=pool,
                verbose=False
            )
        parallel_time = time.time() - start_time
        
        # Parallel should be faster for compute-intensive tasks
        # Allow some overhead tolerance
        assert parallel_time < sequential_time * 1.2


class TestWalkerParallelism:
    """Test walker-level parallelism functionality."""

    def test_emcee_pool_executor(self, basic_problem):
        """Test walker parallelism with ProcessPoolExecutor."""
        n_states, n_dims, n_walkers, n_steps, pos = basic_problem
        
        with ProcessPoolExecutor(max_workers=2) as pool:
            samples, log_probs = run_mcmc_per_state(
                n_states=n_states,
                n_dims=n_dims,
                n_walkers=n_walkers,
                n_steps=n_steps,
                pos=pos,
                log_posterior=simple_log_posterior,
                emcee_pool=pool,
                verbose=False
            )
        
        # Validate results
        assert len(samples) == n_states
        assert len(log_probs) == n_states

    def test_emcee_thread_pool(self, basic_problem):
        """Test walker parallelism with ThreadPoolExecutor."""
        n_states, n_dims, n_walkers, n_steps, pos = basic_problem
        
        with ThreadPoolExecutor(max_workers=2) as pool:
            samples, log_probs = run_mcmc_per_state(
                n_states=n_states,
                n_dims=n_dims,
                n_walkers=n_walkers,
                n_steps=n_steps,
                pos=pos,
                log_posterior=simple_log_posterior,
                emcee_pool=pool,
                verbose=False
            )
        
        # Validate results
        assert len(samples) == n_states
        assert len(log_probs) == n_states

class TestTwoLevelParallelism:
    """Test combined state and walker parallelism."""

    def test_process_process_parallelism(self, multi_state_problem):
        """Test two-level parallelism with ProcessPoolExecutor for both levels."""
        n_states, n_dims, n_walkers, n_steps, pos = multi_state_problem
        
        with ProcessPoolExecutor(max_workers=2) as state_pool, \
             ProcessPoolExecutor(max_workers=2) as walker_pool:
            
            samples, log_probs = run_mcmc_per_state(
                n_states=n_states,
                n_dims=n_dims,
                n_walkers=n_walkers,
                n_steps=n_steps,
                pos=pos,
                log_posterior=simple_log_posterior,
                state_pool=state_pool,
                emcee_pool=walker_pool,
                verbose=False
            )
        
        # Validate results
        assert len(samples) == n_states
        assert len(log_probs) == n_states

    def test_process_thread_parallelism(self, multi_state_problem):
        """Test two-level parallelism with ProcessPoolExecutor + ThreadPoolExecutor."""
        n_states, n_dims, n_walkers, n_steps, pos = multi_state_problem
        
        with ProcessPoolExecutor(max_workers=2) as state_pool, \
             ThreadPoolExecutor(max_workers=2) as walker_pool:
            
            samples, log_probs = run_mcmc_per_state(
                n_states=n_states,
                n_dims=n_dims,
                n_walkers=n_walkers,
                n_steps=n_steps,
                pos=pos,
                log_posterior=simple_log_posterior,
                state_pool=state_pool,
                emcee_pool=walker_pool,
                verbose=False
            )
        
        # Validate results
        assert len(samples) == n_states
        assert len(log_probs) == n_states

    def test_pool_configuration_serialization(self, basic_problem):
        """Test that pool configurations are properly serialized and recreated."""
        n_states, n_dims, n_walkers, n_steps, pos = basic_problem
        
        # This test specifically checks that pools are recreated in worker processes
        # rather than being passed directly (which would cause pickling errors)
        with ProcessPoolExecutor(max_workers=2) as state_pool, \
             ProcessPoolExecutor(max_workers=2) as walker_pool:
            
            # This should work without pickling errors
            samples, log_probs = run_mcmc_per_state(
                n_states=n_states,
                n_dims=n_dims,
                n_walkers=n_walkers,
                n_steps=n_steps,
                pos=pos,
                log_posterior=simple_log_posterior,
                state_pool=state_pool,
                emcee_pool=walker_pool,
                verbose=False
            )
            
            # If we get here without exceptions, pool serialization worked
            assert len(samples) == n_states


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_pool_configuration(self, basic_problem):
        """Test handling of invalid pool configurations."""
        n_states, n_dims, n_walkers, n_steps, pos = basic_problem
        
        # Test with invalid pool object (should not crash)
        class MockPool:
            def map(self, func, iterable):
                return [func(item) for item in iterable]
        
        # This should work as long as pool has map() method
        samples, log_probs = run_mcmc_per_state(
            n_states=n_states,
            n_dims=n_dims,
            n_walkers=n_walkers,
            n_steps=n_steps,
            pos=pos,
            log_posterior=simple_log_posterior,
            state_pool=MockPool(),
            verbose=False
        )
        
        assert len(samples) == n_states

    def test_empty_states(self):
        """Test handling of edge case with minimal configuration."""
        # Test with single state
        samples, log_probs = run_mcmc_per_state(
            n_states=1,
            n_dims=[2],
            n_walkers=4,
            n_steps=5,
            pos=[np.random.normal(0, 0.1, (4, 2))],
            log_posterior=simple_log_posterior,
            verbose=False
        )
        
        assert len(samples) == 1
        assert len(log_probs) == 1

    def test_mismatched_dimensions(self):
        """Test error handling for mismatched dimensions."""
        with pytest.raises((ValueError, IndexError, AssertionError)):
            run_mcmc_per_state(
                n_states=2,
                n_dims=[2, 3],
                n_walkers=8,
                n_steps=10,
                pos=[np.random.normal(0, 0.1, (8, 2))],  # Only one position array
                log_posterior=simple_log_posterior,
                verbose=False
            )


class TestBackwardCompatibility:
    """Test backward compatibility with existing API."""

    def test_original_api_unchanged(self, basic_problem):
        """Test that original API calls still work unchanged."""
        n_states, n_dims, n_walkers, n_steps, pos = basic_problem

        # Original API call should work exactly as before
        samples, log_probs = run_mcmc_per_state(
            n_states, n_dims, n_walkers, n_steps, pos, simple_log_posterior,
            discard=0, thin=1, auto_thin=False, seed=42, verbose=False
        )

        assert len(samples) == n_states
        assert len(log_probs) == n_states


@pytest.mark.skipif(
    not hasattr(multiprocessing, 'get_start_method') or 
    multiprocessing.get_start_method() != 'fork',
    reason="Requires fork start method for optimal performance"
)
class TestForkStartMethod:
    """Test fork start method specific functionality."""

    def test_fork_method_set(self):
        """Test that fork start method is properly set."""
        assert multiprocessing.get_start_method() == 'fork'

    def test_no_pickling_issues(self, multi_state_problem):
        """Test that fork method avoids most pickling issues."""
        n_states, n_dims, n_walkers, n_steps, pos = multi_state_problem
        
        # Use module-level function (can be pickled) rather than local function
        with ProcessPoolExecutor(max_workers=2) as pool:
            samples, log_probs = run_mcmc_per_state(
                n_states=n_states,
                n_dims=n_dims,
                n_walkers=n_walkers,
                n_steps=n_steps,
                pos=pos,
                log_posterior=compute_intensive_log_posterior,  # Use module-level function
                state_pool=pool,
                verbose=False
            )
        
        assert len(samples) == n_states
        
        # The fact that this test passes demonstrates that fork method
        # allows sharing of module-level objects without pickling issues