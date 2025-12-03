# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pyTransC is a Python library implementing Trans-Conceptual MCMC sampling algorithms for Bayesian inference across competing model assumptions. The library provides three main sampling approaches for handling uncertainty across different conceptual states/models.

## Development Commands

### Testing
```bash
python -m pytest tests/
python -m pytest tests/test_pytransc/test_samplers/test_product_space.py  # Run specific test file
```

### Linting and Code Quality
```bash
python -m ruff check                    # Check code style and lint issues
python -m ruff check --fix             # Auto-fix linting issues where possible
python -m ruff format                   # Format code
pre-commit run --all-files             # Run all pre-commit hooks
```

### Installation for Development
```bash
pip install -e .                       # Install in editable mode
pip install -e ".[dev]"                # Install with dev dependencies
pip install -e ".[notebooks]"          # Install with Jupyter dependencies
```

## Architecture

### Core Module Structure
- `src/pytransc/samplers/` - Main sampling algorithms
  - `product_space.py` - Fixed-dimensional sampling over product space of states
  - `state_jump.py` - RJ-MCMC style algorithm with pseudo-prior proposals
  - `ensemble_resampler.py` - Single parameter Metropolis over state indicator
  - `per_state.py` - Independent MCMC sampling within each state
  - `_emcee.py` - Shared emcee integration utilities
  - `_utils.py` - Internal sampler utilities

- `src/pytransc/pseudoprior/` - Pseudo-prior construction
  - `auto_pseudo.py` - Automatic pseudo-prior construction using mixture models
  - `gaussian_mixture.py` - Gaussian mixture model implementations
  - `mean_covariance.py` - Mean-covariance based pseudo-priors

- `src/pytransc/analysis/` - Post-processing and analysis tools
  - `laplace.py` - Marginal likelihood estimation via Laplace approximation
  - `samples.py` - Sample extraction and processing
  - `visits.py` - State visit analysis and acceptance rates
  - `integration.py` - Integration methods for evidence calculation

- `src/pytransc/utils/` - Utility functions and types
  - `types.py` - Type definitions for multi-dimensional arrays and protocols
  - `exceptions.py` - Custom exception classes
  - `autocorr.py` - Autocorrelation analysis utilities
  - `forward_context.py` - Context manager for forward pool parallelization

### Key Concepts
- **Trans-Conceptual Sampling**: Bayesian inference across models with different assumptions/states
- **Pseudo-priors**: Bridge distributions enabling transitions between conceptual states
- **Product Space**: Joint parameter space across all model states
- **State Indicator**: Variable tracking which conceptual model is active
- **Forward Pool**: Parallelization mechanism for forward solver calls within log_posterior functions

### Main Entry Points
All samplers are functions (not classes) that can be imported directly:
```python
from pytransc.samplers import (
    run_product_space_sampler,
    run_state_jump_sampler,
    run_ensemble_resampler,
    run_mcmc_per_state
)
```

Pseudo-prior construction:
```python
from pytransc.pseudoprior import build_auto_pseudo_prior, get_available_pseudo_priors
```

Analysis tools:
```python
from pytransc.analysis import run_laplace_evidence_approximation
```

### Coding Standards
- Uses Ruff for linting with NumPy docstring convention
- Type hints required for function parameters and returns
- Maximum complexity limit of 25 (McCabe)
- Python >=3.11 required
- Dependencies: NumPy, SciPy, scikit-learn, emcee, matplotlib, arviz, tqdm
- Pre-commit hooks enforce code quality (ruff, yamllint, security checks)

### Example Structure
The `examples/` directory contains complete workflows:
- `Regression/` - Polynomial regression across 4 states with all three samplers
- `AirborneEM/` - Real-world geophysical application with airborne electromagnetic data
- `Tomography/` - 2D borehole tomography demonstrating three-level parallelism
- `Gravity/` - Gravity modeling with different geometric assumptions
- Each example includes Jupyter notebooks demonstrating the full analysis pipeline

### Parallelization Architecture
The library supports three levels of parallelism:
1. **Walker-level**: Parallel execution of multiple MCMC walkers (via `walker_pool`)
2. **State-level**: Parallel pseudo-prior evaluation across states (via `state_pool`)
3. **Forward-level**: Parallel forward solver calls within log_posterior (via `forward_pool`)

Forward pools are accessed in log_posterior functions via:
```python
from pytransc.utils.forward_context import get_forward_pool
forward_pool = get_forward_pool()
```