"""
PDF generating functions for Trans-C sampling problem applied to multi-modal posterior.

These functions are defined in a separate module to ensure compatibility
with multiprocessing on macOS and Windows, which use 'spawn' instead of 'fork'.
"""
import numpy as np
rng = np.random.default_rng(42)
import scipy.stats as stats
from utils.IO_utils import (
    read_ensembles,
)
from pytransc.pseudoprior import build_auto_pseudo_prior

def create_log_posterior(dirname,n_components,weights,standardize=False):
    norm_post,ndims = create_norm_log_posterior(dirname,n_components)  # create a normalized posterior using GMM with n_components
    log_posterior = create_unormalized_log_posterior(norm_post,weights) # create unormalized posterior class using weights
    return log_posterior,ndims

def create_norm_log_posterior(dirname,n_components,standardize=False):
    # read in reference ensembles
    ensemble_per_state, log_likelihood_ens, nstates, ndims, nens = read_ensembles(dirname)

    # build GMM
    log_posterior = build_auto_pseudo_prior(
        ensemble_per_state=ensemble_per_state,
        n_components=n_components, 
        standardize=standardize,
        #reg_covar=1e-15,
        )
    return log_posterior,ndims

class create_unormalized_log_posterior: # log posterior wrapper to include imposed weights in each state
    """Class for within state proposal for state jump sampler."""
    rng = np.random.default_rng(42)

    def __init__(self,prop,weights): # prop is list of GaussianMixture classes supplied by the user
        self.prop = prop
        self.weights = weights
    def __call__(self, x: np.ndarray, state: int) -> float:
        """Call method to generate proposal."""
        return self.prop(x,state) + np.log(self.weights[state]) # log density for posterior

    def draw_deviate(self, state: int) -> np.ndarray:
        """Draw from the log posterior distribution."""
        return self.prop.draw_deviate(state)

class log_diag_proposal: # diagonal Gaussian proposal
    """Class for within state proposal for state jump sampler."""
    rng = np.random.default_rng(42)

    def __init__(self,prop): # prop is list of GaussianMixture classes supplied by the user
        self.nstates = len(prop)
        means,std = [],[]
        for i in range(self.nstates):
            means.append(prop[i].mean) # Proposal means for each parameter in state
            std.append(prop[i].covariances_[0]) # Proposal variance for each parameter in state
        self.means = means
        self.std = std
    def __call__(self, x: np.ndarray, state: int) -> float:
        """Call method to generate proposal."""
        return 0.0  # log ratio for symmetric proposal

    def propose(self, x: np.ndarray, state: int) -> np.ndarray:
        """Propose from the proposal distribution."""
        #print(state,len(x),x)
        i = rng.choice(np.arange(len(x)))
        var = self.std[state][i] ** 2
        _x = np.copy(x)
        _x[i] += stats.multivariate_normal.rvs(mean=0.0, cov=var)
        if not isinstance(_x, np.ndarray):
            _x = np.array([_x])  # deal with 1D case which returns a scalar
        return _x

def create_log_diag_proposal(ensemble_per_state,n_components, covariance_type='diag'):
    diag_proposal = build_auto_pseudo_prior(ensemble_per_state=ensemble_per_state,n_components=n_components, covariance_type=covariance_type)
    return log_diag_proposal(diag_proposal.gaussian_mixtures)
