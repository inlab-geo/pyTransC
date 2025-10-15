"""Tests for automatic pseudo-prior construction.

This module tests the rescale_gmm_covariances function and build_auto_pseudo_prior
with various parameter space dimensions, especially focusing on 1D parameter spaces
which have historically been problematic.
"""

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

from pytransc.utils.auto_pseudo import (
    GaussianMixtureStandardizedPseudoPrior,
    build_auto_pseudo_prior,
    rescale_gmm_covariances,
)


class TestRescaleGMMCovariances:
    """Test rescale_gmm_covariances function with various configurations."""

    @pytest.fixture
    def gmm_1d_full(self):
        """Create a fitted 1D GMM with full covariance."""
        data = np.random.randn(100, 1)
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
        gmm.fit(data)
        return gmm

    @pytest.fixture
    def gmm_1d_diag(self):
        """Create a fitted 1D GMM with diagonal covariance."""
        data = np.random.randn(100, 1)
        gmm = GaussianMixture(n_components=2, covariance_type="diag", random_state=42)
        gmm.fit(data)
        return gmm

    @pytest.fixture
    def gmm_1d_spherical(self):
        """Create a fitted 1D GMM with spherical covariance."""
        data = np.random.randn(100, 1)
        gmm = GaussianMixture(
            n_components=2, covariance_type="spherical", random_state=42
        )
        gmm.fit(data)
        return gmm

    @pytest.fixture
    def gmm_1d_tied(self):
        """Create a fitted 1D GMM with tied covariance."""
        data = np.random.randn(100, 1)
        gmm = GaussianMixture(n_components=2, covariance_type="tied", random_state=42)
        gmm.fit(data)
        return gmm

    @pytest.fixture
    def gmm_2d_full(self):
        """Create a fitted 2D GMM with full covariance."""
        data = np.random.randn(100, 2)
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
        gmm.fit(data)
        return gmm

    @pytest.fixture
    def gmm_5d_full(self):
        """Create a fitted 5D GMM with full covariance."""
        data = np.random.randn(100, 5)
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
        gmm.fit(data)
        return gmm

    def test_1d_full_covariance_with_vector_scaling(self, gmm_1d_full):
        """Test 1D parameter space with full covariance accepts (1,) scaling vector."""
        s = np.array([2.0])  # Shape (1,) - correct for 1D
        new_means = np.array([[1.0], [2.0]])

        gmm_rescaled = rescale_gmm_covariances(
            gmm_1d_full, s, new_means=new_means, verbose=False
        )

        assert gmm_rescaled.means_.shape == (2, 1)
        assert gmm_rescaled.covariances_.shape == (2, 1, 1)
        np.testing.assert_array_equal(gmm_rescaled.means_, new_means)

    def test_1d_diag_covariance_with_vector_scaling(self, gmm_1d_diag):
        """Test 1D parameter space with diag covariance accepts (1,) scaling vector."""
        s = np.array([2.0])  # Shape (1,) - correct for 1D
        new_means = np.array([[1.0], [2.0]])

        gmm_rescaled = rescale_gmm_covariances(
            gmm_1d_diag, s, new_means=new_means, verbose=False
        )

        assert gmm_rescaled.means_.shape == (2, 1)
        assert gmm_rescaled.covariances_.shape == (2, 1)
        np.testing.assert_array_equal(gmm_rescaled.means_, new_means)

    def test_1d_tied_covariance_with_vector_scaling(self, gmm_1d_tied):
        """Test 1D parameter space with tied covariance accepts (1,) scaling vector."""
        s = np.array([2.0])  # Shape (1,) - correct for 1D
        new_means = np.array([[1.0], [2.0]])

        gmm_rescaled = rescale_gmm_covariances(
            gmm_1d_tied, s, new_means=new_means, verbose=False
        )

        assert gmm_rescaled.means_.shape == (2, 1)
        assert gmm_rescaled.covariances_.shape == (1, 1)
        np.testing.assert_array_equal(gmm_rescaled.means_, new_means)

    def test_1d_spherical_covariance_with_scalar(self, gmm_1d_spherical):
        """Test 1D parameter space with spherical covariance accepts scalar."""
        s = np.array(2.0)  # 0-d scalar
        new_means = np.array([[1.0], [2.0]])

        gmm_rescaled = rescale_gmm_covariances(
            gmm_1d_spherical, s, new_means=new_means, verbose=False
        )

        assert gmm_rescaled.means_.shape == (2, 1)
        assert gmm_rescaled.covariances_.shape == (2,)
        np.testing.assert_array_equal(gmm_rescaled.means_, new_means)

    def test_1d_spherical_covariance_with_single_element_array(self, gmm_1d_spherical):
        """Test 1D parameter space with spherical covariance accepts (1,) array."""
        s = np.array([2.0])  # Shape (1,) - should be converted to scalar
        new_means = np.array([[1.0], [2.0]])

        gmm_rescaled = rescale_gmm_covariances(
            gmm_1d_spherical, s, new_means=new_means, verbose=False
        )

        assert gmm_rescaled.means_.shape == (2, 1)
        assert gmm_rescaled.covariances_.shape == (2,)
        np.testing.assert_array_equal(gmm_rescaled.means_, new_means)

    def test_2d_full_covariance_with_vector_scaling(self, gmm_2d_full):
        """Test 2D parameter space with full covariance."""
        s = np.array([2.0, 3.0])  # Shape (2,)
        new_means = np.array([[1.0, 2.0], [3.0, 4.0]])

        gmm_rescaled = rescale_gmm_covariances(
            gmm_2d_full, s, new_means=new_means, verbose=False
        )

        assert gmm_rescaled.means_.shape == (2, 2)
        assert gmm_rescaled.covariances_.shape == (2, 2, 2)
        np.testing.assert_array_equal(gmm_rescaled.means_, new_means)

    def test_5d_full_covariance_with_vector_scaling(self, gmm_5d_full):
        """Test 5D parameter space with full covariance."""
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Shape (5,)
        new_means = np.random.randn(2, 5)

        gmm_rescaled = rescale_gmm_covariances(
            gmm_5d_full, s, new_means=new_means, verbose=False
        )

        assert gmm_rescaled.means_.shape == (2, 5)
        assert gmm_rescaled.covariances_.shape == (2, 5, 5)
        np.testing.assert_array_equal(gmm_rescaled.means_, new_means)

    def test_wrong_dimension_raises_error(self, gmm_2d_full):
        """Test that wrong scaling vector dimension raises error."""
        s = np.array([2.0])  # Wrong: should be (2,) for 2D

        with pytest.raises(ValueError, match="length equal to n_features"):
            rescale_gmm_covariances(gmm_2d_full, s)

    def test_wrong_ndim_raises_error(self, gmm_2d_full):
        """Test that 2D scaling array raises error."""
        s = np.array([[2.0, 3.0]])  # Wrong: should be 1D

        with pytest.raises(ValueError, match="must be 1D"):
            rescale_gmm_covariances(gmm_2d_full, s)

    def test_spherical_with_vector_raises_error(self, gmm_1d_spherical):
        """Test that spherical covariance rejects multi-element vector."""
        s = np.array([2.0, 3.0])  # Wrong: spherical needs scalar

        with pytest.raises(ValueError, match="must be a scalar"):
            rescale_gmm_covariances(gmm_1d_spherical, s)

    def test_covariance_scaling_correct_1d(self, gmm_1d_full):
        """Test that covariance scaling is mathematically correct for 1D."""
        s = np.array([2.0])
        original_cov = gmm_1d_full.covariances_.copy()

        gmm_rescaled = rescale_gmm_covariances(gmm_1d_full, s, verbose=False)

        # For full covariance: new_cov = S @ old_cov @ S where S = diag(s)
        # For 1D: S = [[2.0]], so new_cov should be 4 * old_cov
        expected_cov = 4.0 * original_cov
        np.testing.assert_allclose(
            gmm_rescaled.covariances_, expected_cov, rtol=1e-10
        )

    def test_covariance_scaling_correct_2d(self, gmm_2d_full):
        """Test that covariance scaling is mathematically correct for 2D."""
        s = np.array([2.0, 3.0])
        original_cov = gmm_2d_full.covariances_.copy()

        gmm_rescaled = rescale_gmm_covariances(gmm_2d_full, s, verbose=False)

        # For each component, check S @ cov @ S
        S = np.diag(s)
        for k in range(gmm_2d_full.n_components):
            expected_cov_k = S @ original_cov[k] @ S
            np.testing.assert_allclose(
                gmm_rescaled.covariances_[k], expected_cov_k, rtol=1e-10
            )

    def test_precision_cholesky_updated(self, gmm_1d_full):
        """Test that precisions_cholesky_ is correctly recomputed."""
        s = np.array([2.0])

        gmm_rescaled = rescale_gmm_covariances(gmm_1d_full, s, verbose=False)

        # Verify that we can score samples (this uses precisions_cholesky_)
        test_sample = np.array([[0.0]])
        try:
            score = gmm_rescaled.score_samples(test_sample)
            assert np.isfinite(score)
        except Exception as e:
            pytest.fail(
                f"score_samples failed, indicating precisions_cholesky_ not updated: {e}"
            )


class TestGaussianMixtureStandardizedPseudoPrior:
    """Test GaussianMixtureStandardizedPseudoPrior with various dimensions."""

    def test_1d_ensemble(self):
        """Test pseudo-prior construction with 1D ensemble."""
        np.random.seed(42)
        ensemble_1d = [np.random.randn(100, 1)]  # 1D parameter space

        # This should not raise an error
        pseudo_prior = GaussianMixtureStandardizedPseudoPrior(
            ensemble_1d, n_components=2, covariance_type="full", random_state=42
        )

        # Test evaluation
        x = np.array([0.0])
        log_prob = pseudo_prior(x, 0)
        assert np.isfinite(log_prob)

        # Test sampling
        sample = pseudo_prior.draw_deviate(0)
        assert sample.shape == (1,)

    def test_2d_ensemble(self):
        """Test pseudo-prior construction with 2D ensemble."""
        np.random.seed(42)
        ensemble_2d = [np.random.randn(100, 2)]  # 2D parameter space

        pseudo_prior = GaussianMixtureStandardizedPseudoPrior(
            ensemble_2d, n_components=2, covariance_type="full", random_state=42
        )

        # Test evaluation
        x = np.array([0.0, 0.0])
        log_prob = pseudo_prior(x, 0)
        assert np.isfinite(log_prob)

        # Test sampling
        sample = pseudo_prior.draw_deviate(0)
        assert sample.shape == (2,)

    def test_mixed_dimensions(self):
        """Test pseudo-prior with states of different dimensions."""
        np.random.seed(42)
        ensemble_per_state = [
            np.random.randn(100, 1),  # State 0: 1D
            np.random.randn(100, 2),  # State 1: 2D
            np.random.randn(100, 3),  # State 2: 3D
        ]

        pseudo_prior = GaussianMixtureStandardizedPseudoPrior(
            ensemble_per_state, n_components=2, covariance_type="full", random_state=42
        )

        # Test each state
        for state in range(3):
            ndim = state + 1
            x = np.zeros(ndim)
            log_prob = pseudo_prior(x, state)
            assert np.isfinite(log_prob)

            sample = pseudo_prior.draw_deviate(state)
            assert sample.shape == (ndim,)

    def test_all_covariance_types_1d(self):
        """Test all covariance types work with 1D ensemble."""
        np.random.seed(42)
        ensemble_1d = [np.random.randn(100, 1)]

        for cov_type in ["full", "diag", "tied", "spherical"]:
            pseudo_prior = GaussianMixtureStandardizedPseudoPrior(
                ensemble_1d, n_components=2, covariance_type=cov_type, random_state=42
            )

            x = np.array([0.0])
            log_prob = pseudo_prior(x, 0)
            assert np.isfinite(
                log_prob
            ), f"Failed for covariance_type={cov_type} with 1D"


class TestBuildAutoPseudoPrior:
    """Test build_auto_pseudo_prior function."""

    def test_build_with_1d_ensemble(self):
        """Test building auto pseudo-prior with 1D ensemble."""
        np.random.seed(42)
        ensemble_per_state = [
            np.random.randn(100, 1),  # State 0: 1D
            np.random.randn(100, 2),  # State 1: 2D
        ]

        pseudo_prior = build_auto_pseudo_prior(ensemble_per_state=ensemble_per_state)

        # Test evaluation for both states
        x0 = np.array([0.0])
        log_prob0 = pseudo_prior(x0, 0)
        assert np.isfinite(log_prob0)

        x1 = np.array([0.0, 0.0])
        log_prob1 = pseudo_prior(x1, 1)
        assert np.isfinite(log_prob1)

    def test_build_with_regression_like_ensemble(self):
        """Test with ensemble structure similar to regression example."""
        np.random.seed(42)
        # Simulate the regression example: polynomial orders 0, 1, 2, 3
        ensemble_per_state = [
            np.random.randn(1000, 1),  # Order 0: 1 parameter
            np.random.randn(1000, 2),  # Order 1: 2 parameters
            np.random.randn(1000, 3),  # Order 2: 3 parameters
            np.random.randn(1000, 4),  # Order 3: 4 parameters
        ]

        # This was failing before the fix
        pseudo_prior = build_auto_pseudo_prior(ensemble_per_state=ensemble_per_state)

        # Test evaluation for all states
        for state in range(4):
            ndim = state + 1
            x = np.zeros(ndim)
            log_prob = pseudo_prior(x, state)
            assert np.isfinite(
                log_prob
            ), f"Failed for state {state} with {ndim} dimensions"

    def test_evaluation_on_ensemble_samples(self):
        """Test pseudo-prior evaluation on samples from the ensemble."""
        np.random.seed(42)
        ensemble_per_state = [
            np.random.randn(100, 1),  # 1D
        ]

        pseudo_prior = build_auto_pseudo_prior(ensemble_per_state=ensemble_per_state)

        # Evaluate on actual ensemble samples
        log_probs = []
        for x in ensemble_per_state[0][:10]:  # Test first 10 samples
            log_prob = pseudo_prior(x, 0)
            assert np.isfinite(log_prob)
            log_probs.append(log_prob)

        # All log probabilities should be finite and reasonable
        assert all(np.isfinite(log_probs))
        # They should be negative (log probabilities)
        assert all(lp < 0 for lp in log_probs)
