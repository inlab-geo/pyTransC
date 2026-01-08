"""Utility functions for reading and processing input ensemles."""

import arviz as az
import numpy as np

def read_chain(fp):
    """TODO: add docstring."""
    fh = open(fp)
    lines = fh.readlines()
    chain = []
    for line in lines:
        fields = line.split()
        msft = float(fields[0])
        n = int(fields[1])
        d = np.array(fields[2 : n + 2], dtype=float)
        c = np.array(fields[n + 2 : n + 2 + n], dtype=float)
        chain.append([n, msft, d, c])
    return chain


# read in enembles
def read_ensembles(dir):
    """TODO: add docstring."""
    log_posterior_ens, ensemble_per_state, ndims = [], [], []
    for i in range(10):
        chain = read_chain(dir + f"/lyr{i + 1:02d}/chain.out")
        n, msft, d, logc = chain[0]
        ndims.append(len(np.concatenate((d, logc))) - 1)
        y = []
        x = np.zeros((len(chain), ndims[i]))
        for j, piece in enumerate(chain):
            n, msft, d, logc = piece
            y.append(-0.5 * msft)
            x[j] = np.concatenate((d[1:], logc))
        log_posterior_ens.append(y)
        ensemble_per_state.append(x)

    nstates = len(log_posterior_ens)
    nens = len(chain)
    return ensemble_per_state, log_posterior_ens, nstates, ndims, nens


def read_ensembles_thicknesses():  # not yet implemented
    """TODO: add docstring."""
    log_posterior_ens, ensemble_per_state, ndims = [], [], []
    for i in range(10):
        chain = read_chain(f"lyr{i + 1:02d}/chain.out")
        msft, n, d, logc = chain[0]
        ndims.append(len(np.concatenate((d, logc))) - 1)
        y = []
        x = np.zeros((len(chain), ndims[i]))
        for j, piece in enumerate(chain):
            msft, n, d, logc = piece
            y.append(msft)
            x[j] = np.concatenate((d[1:], logc))
        log_posterior_ens.append(y)
        ensemble_per_state.append(x)

    nstates = len(log_posterior_ens)
    nens = len(chain)
    return ensemble_per_state, log_posterior_ens, nstates, ndims, nens


def get_names(ndim, state, verbose=False):
    """TODO: add docstring."""
    ncond = ndim - state
    ndepths = state
    var_names = []
    d_names = [f"depth {i + 1}" for i in range(ndepths)]
    c_names = [f"log cond {i + 1}" for i in range(ncond)]
    var_names = []
    for x in range(state):
        var_names.append(d_names[x])
    for x in range(state):
        var_names.append(c_names[x])
    var_names.append(c_names[-1])
    if verbose:
        print(var_names)
    return var_names


def create_InferenceData_object(
    ensemble, log_posterior, log_likelihood=None, variable_names=None
):
    """TODO: add docstring."""
    nstates = len(ensemble)
    azobjs = []
    for state in range(nstates):
        ndim = ensemble[state].shape[-1]
        samples = ensemble[state]
        # if(variable_names is None):
        # var_names = ["param{}".format(i) for i in range(ndim)]
        # else:
        var_names = get_names(ndim, state)
        if samples.ndim == 2:
            samples_dict = {name: samples[:, i] for i, name in enumerate(var_names)}
        else:
            samples_dict = {name: samples[:, :, i] for i, name in enumerate(var_names)}

        log_likelihood_dict = {}
        if log_likelihood is not None:
            log_likelihood_dict = {"log_likelihood": log_likelihood[state]}
        log_posterior_dict = {"log_posterior": log_posterior[state]}

        # convert to arviz.InferenceData
        azobjs.append(
            az.from_dict(
                posterior=samples_dict,
                sample_stats=log_posterior_dict,
                log_likelihood=log_likelihood_dict,
            )
        )
    return azobjs


def create_InferenceData_object_per_state(
    ensemble, log_posterior, ndim, state, log_likelihood=None
):
    """TODO: add docstring."""
    samples = ensemble
    var_names = get_names(ndim, state)
    if samples.ndim == 2:
        samples_dict = {name: samples[:, i] for i, name in enumerate(var_names)}
    else:
        samples_dict = {name: samples[:, :, i] for i, name in enumerate(var_names)}

    log_likelihood_dict = {}
    if log_likelihood is not None:
        log_likelihood_dict = {"log_likelihood": log_likelihood[state]}
    log_posterior_dict = {"log_posterior": log_posterior[state]}

    return az.from_dict(
        posterior=samples_dict,  # convert to arviz.InferenceData
        sample_stats=log_posterior_dict,
        log_likelihood=log_likelihood_dict,
    )


def ens_diagnostics(iss, elapsed_time, thin=15, discard=0):
    """TODO: add docstring."""
    print("\n Algorithm type                                      :", iss.alg)
    print(" Number of walkers                                   :", iss.nwalkers)
    print(" Number of steps                                     :", iss.nsteps)
    # print(' Acceptance rates for walkers within states:  \n',*iss.accept_within_per_walker,'\n')
    # print(' Acceptance rates for walkers between states: \n',*iss.accept_between_per_walker,'\n')
    # print(' Average % acceptance rate for within states         :',np.round(istomo_ens.accept_within,2))
    print(
        " Average % acceptance rate for between states        :",
        np.round(iss.accept_between, 2),
    )
    # extract trans-D samples and chains
    # discard = 0                  # chain burnin
    # thin = 15                    # chain thinning
    chain, states_chain = iss.get_visits_to_states(
        discard=discard,
        thin=thin,
        normalize=True,
        walker_average="median",
        return_samples=True,
    )
    print(
        " Auto correlation time for between state sampling    :",
        np.round(iss.autocorr_time_for_between_state_jumps, 5),
    )
    print(
        " Total number of state changes for all walkers       :",
        iss.total_state_changes,
    )
    print(
        " Estimated relative evidences ens                    :",
        *np.round((iss.relative_marginal_likelihoods), 5),
    )
    print(
        " Elapsed time                                        :",
        np.round(elapsed_time, 2),
        "s \n",
    )
    return chain, states_chain


# set proposals only between neighbouring states (kneighbours is the maximum diagonal to allow a transition)
def setproposalweights(nstates, kneighbours):
    """TODO: add docstring."""
    A = np.ones((nstates, nstates))
    A[np.diag_indices(nstates)] = 0.0
    if kneighbours == -1:
        return A  # set transition probabilities for all states equally
    A[np.triu_indices_from(A, k=kneighbours + 1)] = 0.0
    A[np.tril_indices_from(A, k=-(kneighbours + 1))] = 0.0
    return A  # set transition probabilities between selected states only
