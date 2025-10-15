"""
Trans conceptual sampling across borehole tomography models

This script demonstrates MPI-based parallelization for trans-conceptual MCMC
sampling applied to seismic travel time tomography.

Setup Instructions (Fedora):
-----------------------------
1. Install OpenMPI:
   sudo dnf install openmpi openmpi-devel

2. Load OpenMPI module (needs to be done each session or add to ~/.bashrc):
   module load mpi/openmpi-x86_64

   Or manually add to PATH:
   export PATH=$PATH:/usr/lib64/openmpi/bin
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/openmpi/lib

3. Install Python MPI packages in your virtual environment:
   pip install mpi4py schwimmbad

4. Verify installation:
   which mpiexec
   python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_size())"

MPI Execution Model (Master/Worker Pattern):
---------------------------------------------
When you run `mpiexec -n 16 python script.py`, ALL 16 processes execute the
entire script from top to bottom. To avoid chaos, we use the Master/Worker pattern:

- **Master process**: Coordinates work and calls sampling functions
- **Worker processes**: Block at `pool.wait()`, execute tasks from master, then exit

The pattern looks like:
    if not state_pool.is_master():
        state_pool.wait()  # Workers wait here for tasks from master
        sys.exit(0)        # When done, exit - don't run code below

    # Only master continues here to coordinate work

This ensures only ONE process coordinates the work while others serve as workers.
In our nested hierarchy:
- 1 state master coordinates 2 states across 16 processes
- 2 walker masters (1 per state) coordinate walkers
- 4 forward masters (1 per walker) coordinate forward computations

Uses NESTED MPI communicators for 3-level parallelism:
State -> Walker (nested in State) -> Forward (nested in Walker)

Pool Structure:
- State Pool:   2 groups × 8 processes each = 16 total processes
- Walker Pool:  4 groups × 4 processes each = 16 total processes (2 per state)
- Forward Pool: 8 groups × 2 processes each = 16 total processes (2 per walker)

Note on Process Count:
The equivalent ProcessPoolExecutor approach in the notebook uses:
- 2 state workers + 4 walker workers + 8 forward workers = 14 workers + 1 main = 15 total
This MPI version uses 16 processes for cleaner power-of-2 splits in the nested hierarchy.
The extra process allows symmetric subdivision at each level (8→4→2 vs awkward 7→3.5→...)

Performance Note:
-----------------
This example uses a SMALL problem size (nsteps=5) for demonstration purposes.
With such small problems, MPI overhead (process initialization, communication,
synchronization) DOMINATES the actual computation time, making MPI appear slower
than ProcessPoolExecutor.

For PRODUCTION use, increase nsteps to 5000+ where:
- Actual computation dominates overhead
- MPI's superior parallelism (16 vs 15 processes) shows significant speedup
- Nested MPI communicators provide better load balancing

Rule of thumb: MPI excels when computation time >> communication overhead

MPI Process Mapping (each process belongs to ONE group in each pool):
┌────────────┬─────────────┬──────────────┬───────────────┐
│ Process ID │ State Pool  │ Walker Pool  │ Forward Pool  │
│  (Global)  │ Rank (local)│ Rank (local) │ Rank (local)  │
├────────────┼─────────────┼──────────────┼───────────────┤
│     0      │      0      │      0       │      0        │
│     1      │      1      │      1       │      1        │
│     2      │      2      │      2       │      0        │
│     3      │      3      │      3       │      1        │
│     4      │      4      │      0       │      0        │
│     5      │      5      │      1       │      1        │
│     6      │      6      │      2       │      0        │
│     7      │      7      │      3       │      1        │
│     8      │      0      │      0       │      0        │
│     9      │      1      │      1       │      1        │
│    10      │      2      │      2       │      0        │
│    11      │      3      │      3       │      1        │
│    12      │      4      │      0       │      0        │
│    13      │      5      │      1       │      1        │
│    14      │      6      │      2       │      0        │
│    15      │      7      │      3       │      1        │
└────────────┴─────────────┴──────────────┴───────────────┘

Nested Hierarchy:
- State Pool 0 (procs 0-7):
  - Walker Pool 0 (procs 0-3):
    - Forward Pool 0 (procs 0-1)
    - Forward Pool 1 (procs 2-3)
  - Walker Pool 1 (procs 4-7):
    - Forward Pool 2 (procs 4-5)
    - Forward Pool 3 (procs 6-7)
- State Pool 1 (procs 8-15):
  - Walker Pool 2 (procs 8-11):
    - Forward Pool 4 (procs 8-9)
    - Forward Pool 5 (procs 10-11)
  - Walker Pool 3 (procs 12-15):
    - Forward Pool 6 (procs 12-13)
    - Forward Pool 7 (procs 14-15)

Usage:
    mpiexec -n 16 python FMM_TransC_borehole_parallel_mpi.py

"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import pyfm2d
import time
from tqdm import tqdm
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import scipy.stats as stats
from functools import partial
from sklearn.mixture import GaussianMixture

from schwimmbad import MPIPool
from mpi4py import MPI


# pyTransC imports
from pytransc.samplers import run_mcmc_per_state, run_ensemble_resampler
from pytransc.utils.types import FloatArray
from pytransc.utils.auto_pseudo import build_auto_pseudo_prior
from pytransc.analysis.visits import get_visits_to_states


# ============================================================================
# Utility routines for building seismic velocity models
# ============================================================================

def get_gauss_model(extent, nx, ny, factor=1.):
    """Build two gaussian anomaly velocity model"""
    vc1 = 1700*factor                           # velocity of circle 1 in m/s
    vc2 = 2300*factor                           # velocity of circle 2 in m/s
    vb = 2000*factor                            # background velocity
    dx = (extent[1]-extent[0])/nx               # cell width
    dy = (extent[3]-extent[2])/ny               # cell height
    xc = np.linspace(extent[0], extent[1], nx)   # cell centre
    yc = np.linspace(extent[2], extent[3], ny)   # cell centre
    X, Y = np.meshgrid(xc, yc, indexing='ij')    # cell centre mesh

    # Multivariate Normal
    dex = extent[1]-extent[0]
    dey = extent[3]-extent[2]
    c1x = extent[0] + (7.0-extent[0])*dex/20.
    c2x = extent[0] + (12.0-extent[0])*dex/20.
    c1y = extent[0] + (22.0-extent[0])*dey/30.
    c2y = extent[0] + (10.0-extent[0])*dey/30.
    s1 = 6.0*dex/20.
    s2 = 10.0*dex/20.
    c1, sig1 = np.array([c1x, c1y])*factor, s1*(factor**2)
    c2, sig2 = np.array([c2x, c2y])*factor, s2*(factor**2)
    rv1 = multivariate_normal(c1, [[sig1, 0], [0, sig1]])
    rv2 = multivariate_normal(c2, [[sig2, 0], [0, sig2]])

    # Probability Density
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    gauss1, gauss2 = rv1.pdf(pos), rv2.pdf(pos)
    return vb*np.ones([nx, ny]) + (vc1-vb)*gauss1/np.max(gauss1) + (vc2-vb)*gauss2/np.max(gauss2)


def generate_covariance_matrix_inv(model_shape: tuple, corr_lengths: tuple, sigma: float):
    """Gaussian inverse covariance matrix implementation from Juerg Hauser"""
    # ensure model_shape and corr_lengths have the same length
    if len(model_shape) != len(corr_lengths):
        raise ValueError(
            "`model_shape` and `corr_lengths` should have the same lengths, "
            f"but got {len(model_shape)} and {len(corr_lengths)}"
        )
    # generate grid of points for each dimension
    grids = np.meshgrid(*[np.arange(dim) for dim in model_shape], indexing="ij")
    # calculate distances between points for each pair of dimensions
    d_squared = sum(
        [
            (grid.ravel()[None, :] - grid.ravel()[:, None]) ** 2 / corr_length**2
            for grid, corr_length in zip(grids, corr_lengths)
        ]
    )
    # construct correlation matrix
    Cp = np.exp(-np.sqrt(d_squared))
    # construct variance matrix
    Sc = np.zeros((np.prod(model_shape), np.prod(model_shape)))
    np.fill_diagonal(Sc, sigma)
    # calculate covariance matrix
    covariance_matrix = Sc @ Cp @ Sc
    # calculate inverse covariance matrix
    covariance_matrix_inv = np.linalg.inv(covariance_matrix)
    return covariance_matrix_inv, covariance_matrix


# ============================================================================
# Initialize MPI early so we can use rank checks for printing
# ============================================================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Only master process prints and creates figures
is_master = (rank == 0)

# ============================================================================
# Build high resolution reference velocity model and calculate travel times
# ============================================================================

if is_master:
    print("Building reference velocity model...")
extent = [0.0, 20.0, 0.0, 30.0]
m = get_gauss_model(extent, 32, 48)
gtrue = pyfm2d.BasisModel(m, extent=extent)
velocity = gtrue.get_velocity()

if is_master:
    fig = plt.figure(figsize=(6, 6))
    pyfm2d.display_model(velocity, extent=extent, figsize=(6, 6), clim=(1700, 2300))
    plt.savefig('reference_velocity_model.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reference_velocity_model.png")

# Build sources and receivers
recs = pyfm2d.wavetracker.generate_surface_points(
    8, extent=extent, surface=[False, True, False, False], addCorners=False
)
srcs = pyfm2d.wavetracker.generate_surface_points(
    16, extent=extent, surface=[True, False, False, False], addCorners=False
)
nr = len(recs)
ns = len(srcs)

# Run wave front tracker to get true times in reference model
options = pyfm2d.WaveTrackerOptions(
    times=True,
    frechet=False,
    paths=True,
    cartesian=True,
)

start = time.time()
result = pyfm2d.calc_wavefronts(velocity, recs, srcs, extent=extent, options=options)
end = time.time()
if is_master:
    print(f"Wall time: {end - start:.2f}s")
    print(f"Number of paths calculated = {len(result.paths)}")
    print(f"Number of travel times calculated = {len(result.ttimes)}")

# Display continuous model and raypaths
if is_master:
    fig = plt.figure(figsize=(6, 6))
    pyfm2d.display_model(
        velocity,
        extent=extent,
        figsize=(6, 6),
        clim=(1700, 2300),
        paths=result.paths,
        recs=recs,
        srcs=srcs,
        alpha=0.1,
    )
    plt.savefig('velocity_model_with_raypaths.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: velocity_model_with_raypaths.png")

# Add noise to travel times
sigma = 0.0001
np.random.seed(61254557)
ttrue = result.ttimes
tobs = ttrue + sigma*np.random.randn(len(ttrue))
if is_master:
    print(f'Noise is {sigma/np.std(ttrue)*100:.2f}% of travel times standard deviation')
Cdinv = np.eye(len(tobs))/(sigma**2)
Cd = np.eye(len(tobs))*sigma**2


# ============================================================================
# Define model states
# ============================================================================

if is_master:
    print("\nDefining model states...")
stateparams = {}
nstates = 4

stateparams['nx'] = [7, 6, 5, 4]
stateparams['ny'] = [9, 8, 7, 6]
corrkm = 2
stateparams['corrx'] = [1, 1, 1, 1]
stateparams['corry'] = [1, 1, 1, 1]
sigma_slowness = 2.5E-6

Cm_inv, Cm, sref, rvs, gs, corrxl, corryl = [None] * nstates, [None] * nstates, [None] * nstates, [None] * nstates, [None] * nstates, [None] * nstates, [None] * nstates

for state in range(nstates):
    nx, ny = stateparams['nx'][state], stateparams['ny'][state]
    corrx = corrkm*nx/(extent[1]-extent[0])
    corry = corrkm*ny/(extent[3]-extent[2])
    Cm_inv[state], Cm[state] = generate_covariance_matrix_inv((nx, ny), (corrx, corry), sigma_slowness)
    gs[state] = pyfm2d.BasisModel(get_gauss_model(extent, nx, ny), extent=extent)
    sref[state] = np.ones([nx, ny]).flatten()/2000.
    Cm[state] = sigma_slowness*np.eye(nx*ny)
    rvs[state] = stats.multivariate_normal(mean=sref[state], cov=Cm[state])
    corrxl[state] = corrx
    corryl[state] = corry

stateparams['Cm'] = Cm
stateparams['Cm_inv'] = Cm_inv
stateparams['sref'] = sref
stateparams['g'] = gs
stateparams['rv'] = rvs
stateparams['corrx'][state] = corrxl
stateparams['corry'][state] = corryl

# Plot the true model projected onto the parameterization in each state
if is_master:
    fig, axes_2d = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes_2d.flatten()
    for state in range(nstates):
        g = stateparams['g'][state]
        pyfm2d.display_model(g.get_velocity(), extent=extent, ax=axes[state], clim=(1700, 2300), diced=False)
    plt.savefig('true_model_all_states.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: true_model_all_states.png")


# ============================================================================
# Log Likelihood and prior for each state
# ============================================================================

def _log_prior(x, state, stateparams):
    rv = stateparams['rv'][state]
    return rv.logpdf(x)

log_prior = partial(_log_prior, stateparams=stateparams)


def _log_likelihood(x, state, tobs, Cdinv, recs, srcs, extent, options, stateparams):
    nx, ny = stateparams['nx'][state], stateparams['ny'][state]
    velocity = 1./x.reshape((nx, ny))
    result = pyfm2d.calc_wavefronts(velocity, recs, srcs, extent=extent, options=options)
    res = tobs - result.ttimes
    LL = -0.5 * res.T @ Cdinv @ res
    return LL

log_likelihood = partial(_log_likelihood, tobs=tobs, Cdinv=Cdinv, recs=recs, srcs=srcs, extent=extent, options=options, stateparams=stateparams)


def log_posterior(x, state):
    lp = log_prior(x, state)
    ll = log_likelihood(x, state)
    return ll + lp


def _log_likelihood_with_forward_pool(x, state, tobs, Cdinv, recs, srcs, extent, options, stateparams):
    from pytransc.utils.forward_context import get_forward_pool
    forward_pool = get_forward_pool()
    nx, ny = stateparams['nx'][state], stateparams['ny'][state]
    velocity = 1./x.reshape((nx, ny))
    result = pyfm2d.calc_wavefronts(velocity, recs, srcs, pool=forward_pool, extent=extent, options=options)
    res = tobs - result.ttimes
    LL = -0.5 * res.T @ Cdinv @ res
    return LL

log_likelihood_with_forward_pool = partial(_log_likelihood_with_forward_pool, tobs=tobs, Cdinv=Cdinv, recs=recs, srcs=srcs, extent=extent, options=options, stateparams=stateparams)


def log_posterior_with_forward_pool(x, state):
    lp = log_prior(x, state)
    ll = log_likelihood_with_forward_pool(x, state)
    return ll + lp


# ============================================================================
# Starting models for random chains
# ============================================================================

if is_master:
    print("\nGenerating starting models...")
ndims = [stateparams['nx'][i]*stateparams['ny'][i] for i in range(nstates)]
nwalkers = 128
nsteps = 5

pos = []
posll = []
for state in range(nstates):
    nx, ny = stateparams['nx'][state], stateparams['ny'][state]
    rv = stats.multivariate_normal(mean=np.zeros(nx*ny), cov=np.eye(nx*ny))
    slow = 1./get_gauss_model(extent, nx, ny).flatten() + sigma_slowness*rv.rvs(size=nwalkers)
    pos.append(slow)
    lp = log_prior(slow[0], state)
    ll = log_likelihood(slow[0], state)
    lpos = log_posterior(slow[0], state)
    if is_master:
        print(f' State {state}: log-prior {lp:.2f}, log-Likelihood {ll:.2f}, lpost {lpos:.2f}')
    posll.append(log_likelihood(pos[state][state], state))

# Plot the starting models
if is_master:
    fig, axes_2d = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes_2d.flatten()
    for state in range(nstates):
        nx, ny = stateparams['nx'][state], stateparams['ny'][state]
        pyfm2d.display_model(1./pos[state][0].reshape((nx, ny)), extent=extent, ax=axes[state], clim=(1700, 2300))
    plt.savefig('starting_models_all_states.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: starting_models_all_states.png")



# ============================================================================
# State, walker and forward parallelism with MPI
# ============================================================================

if is_master:
    print("\n" + "="*60)
    print("THREE NESTED MPI POOLS (State -> Walker -> Forward)")
    print("="*60)
    print("16 processes in nested hierarchy: 2 states × 2 walkers × 2 forward")

# Verify we have exactly 16 processes
if size != 16:
    if is_master:
        print(f"ERROR: This configuration requires exactly 16 MPI processes")
        print(f"       You provided {size} processes")
        print(f"       Run with: mpiexec -n 16 python {sys.argv[0]}")
    sys.exit(1)

# Create NESTED communicator hierarchy:
# State -> Walker (nested in State) -> Forward (nested in Walker)

# Level 1: State communicator - split into 2 groups (each with 8 processes)
state_color = rank // 8  # Ranks 0-7 -> color 0, ranks 8-15 -> color 1
state_comm = comm.Split(state_color, rank)

# Level 2: Walker communicator - subdivide each state group into 2 walker groups
# This splits state_comm (not global comm), creating 2 walker pools per state (4 total)
walker_color = (rank % 8) // 4  # Within each state: 0-3 -> color 0, 4-7 -> color 1
walker_comm = state_comm.Split(walker_color, rank)

# Level 3: Forward communicator - subdivide each walker group into 2 forward groups
# This splits walker_comm, creating 2 forward pools per walker (8 total)
forward_color = (rank % 4) // 2  # Within each walker: 0-1 -> color 0, 2-3 -> color 1
forward_comm = walker_comm.Split(forward_color, rank)

if rank == 0:
    print(f"MPI communicators created (nested hierarchy):")
    print(f"  State pool:   2 groups × 8 processes = 16 total")
    print(f"  Walker pool:  4 groups × 4 processes = 16 total (nested in state)")
    print(f"  Forward pool: 8 groups × 2 processes = 16 total (nested in walker)")
    print(f"  Note: All 16 processes participate in nested hierarchy")

start_time = time.time()

# Create MPIPool for each level with the appropriate communicator
with MPIPool(comm=state_comm) as state_pool, \
     MPIPool(comm=walker_comm) as walker_pool, \
     MPIPool(comm=forward_comm) as forward_pool:

    # Only the master of the top-level state_pool runs the sampling
    if not state_pool.is_master():
        state_pool.wait()
        sys.exit(0)

    if rank == 0:
        print("\nAll three MPI pools created successfully")
        print("Starting sampling...")

    ensembles_three, log_probs_three = run_mcmc_per_state(
        n_states=nstates,
        n_dims=ndims,
        n_walkers=nwalkers,
        n_steps=nsteps,
        pos=pos,
        log_posterior=log_posterior_with_forward_pool,
        verbose=is_master,  # Only master process shows progress
        skip_initial_state_check=True,
        state_pool=state_pool,
        emcee_pool=walker_pool,
        forward_pool=forward_pool,
    )

    three_level_time = time.time() - start_time
    if is_master:
        print(f"\nThree level parallel execution completed in {three_level_time:.2f} seconds")
        print(f"Sample shapes: {[ens.shape for ens in ensembles_three]}")
        print(f"Log prob shapes: {[lp.shape for lp in log_probs_three]}")


