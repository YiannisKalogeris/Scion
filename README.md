# Simulacrum

A modular computational mechanics library with Python orchestration and optional Rust/C++ acceleration. Designed for research in numerical methods for PDEs, scientific machine learning, and uncertainty quantification.

## Project Structure

```
Simulacrum/
├── src/simulacrum/              # The library (reusable across projects)
│   ├── core/                    #   Domain, Field, DOF, logging abstractions
│   ├── mesh/                    #   Mesh generation (built-in + gmsh), import via meshio
│   ├── discretization/          #   Numerical schemes
│   │   ├── fem/                 #     Finite Element Method
│   │   ├── fdm/                 #     Finite Difference Method
│   │   ├── fvm/                 #     Finite Volume Method (scaffold)
│   │   └── sph/                 #     Smoothed Particle Hydrodynamics (scaffold)
│   ├── materials/               #   Constitutive models (linear elastic, thermal, hyperelastic)
│   ├── solvers/                 #   Linear (direct, Krylov) and nonlinear (Newton-Raphson)
│   ├── boundary/                #   Dirichlet / Neumann BC handling
│   ├── geometry/                #   Shape generation, SDF rendering, harmonic maps
│   ├── ml/                      #   Machine learning (MLP, GNN, U-Net, training, surrogates)
│   ├── uq/                      #   Uncertainty quantification (sampling, MCMC, PCE, GP)
│   ├── optimization/            #   Optimisation (gradient, CMA-ES, Bayesian, OED)
│   ├── io/                      #   VTK/VTU output via meshio
│   ├── postprocessing/          #   PyVista-based visualisation
│   └── _backends/               #   Native acceleration
│       ├── rust/                #     PyO3/maturin Rust extension
│       └── cpp/                 #     pybind11/Eigen C++ extension
│
├── projects/                    # Research projects (consumers of the library)
│   └── nsf_domain_mapping/      #   Neural Spline Flow domain-to-square mapping
│       ├── models/              #     NSF architecture
│       ├── config.py            #     Hyperparameters and settings
│       ├── dataset.py           #     Data generation pipeline
│       └── Main_nsf.py          #     Full training/evaluation pipeline
│
├── tests/                       # Unit tests for the library
├── examples/                    # Short self-contained demos
│   └── fem/                     #   e.g. poisson_2d.py
├── benchmarks/                  # Performance benchmarks
├── docs/                        # Documentation source
├── pyproject.toml               # Package configuration and dependencies
└── README.md
```

## Design Principles

- **Python first**: All orchestration in Python. NumPy/SciPy for numerics.
- **Swappable discretizations**: FEM, FDM, FVM, SPH share the same `Domain → LinearSystem → Solver` pipeline.
- **Library vs. projects**: Reusable components live in `src/simulacrum/`; research workflows live in `projects/`.
- **Compiled backends optional**: Rust (via maturin/PyO3) and C++ (via pybind11/Eigen) for hot loops. The library works without them.
- **Meshing is external**: Use gmsh or meshio for mesh generation and import.
- **VTK output**: Write `.vtu` files and visualise in ParaView or interactively with PyVista.

## Quick Start

```bash
# Install core library in development mode
pip install -e ".[dev]"

# Install with JAX support (for NSF project)
pip install -e ".[jax,dev]"

# Install with PyTorch support (for ML subpackage)
pip install -e ".[ml,dev]"

# Install everything
pip install -e ".[all]"

# Run tests
pytest tests/

# Run an example
python examples/fem/poisson_2d.py

# Run the NSF domain mapping project
cd projects/nsf_domain_mapping
python Main_nsf.py
```

## Subpackage Overview

### Core Simulation

| Subpackage | Purpose |
|---|---|
| `core` | Domain, Field, DOF management, logging |
| `mesh` | Mesh generation and import (structured, unstructured, gmsh) |
| `discretization` | FEM, FDM, FVM, SPH assemblers |
| `materials` | Linear elastic, thermal, hyperelastic constitutive models |
| `solvers` | Direct (SciPy), Krylov (CG, GMRES), Newton-Raphson |
| `boundary` | Dirichlet and Neumann boundary condition application |
| `io` | VTK/VTU file output via meshio |
| `postprocessing` | Interactive 3D visualisation with PyVista |

### Geometry

| Module | Purpose |
|---|---|
| `geometry.shapes` | Parameterized closed curves (Ellipse, Superellipse, FourierBlob, StarPolygon, SmoothPolygon) |
| `geometry.sdf` | Signed distance field computation, rendering, and point sampling |
| `geometry.harmonic_maps` | Harmonic map solver for domain-to-square ground truth |

### Machine Learning

| Module | Purpose |
|---|---|
| `ml.networks` | MLP, GNN (mesh-native), U-Net (structured grids) |
| `ml.training` | Generic Trainer with early stopping, physics-informed losses |
| `ml.surrogates` | PDE surrogate wrapper for fast inference |

### Uncertainty Quantification

| Module | Purpose |
|---|---|
| `uq.sampling` | Monte Carlo, Latin Hypercube, Sobol sequences |
| `uq.bayesian` | Metropolis-Hastings, Adaptive MCMC, RealNVP normalizing flows |
| `uq.pce` | Polynomial Chaos Expansions with Sobol sensitivity indices |
| `uq.gp` | Gaussian Process regression / emulation |

### Optimisation

| Module | Purpose |
|---|---|
| `optimization.gradient` | SciPy wrapper (L-BFGS-B, SLSQP) |
| `optimization.evolutionary` | CMA-ES derivative-free optimiser |
| `optimization.bayesian_opt` | Bayesian optimisation with GP surrogates |
| `optimization.oed` | Optimal Experimental Design (A/D/E-optimality) |

## Optional Dependencies

```toml
[project.optional-dependencies]
ml   = ["torch>=2.0"]
jax  = ["jax>=0.4", "jaxlib>=0.4", "equinox>=0.11", "optax>=0.2", "diffrax>=0.5"]
meshing = ["gmsh>=4.11"]
dev  = ["pytest>=7.0", "pytest-benchmark", "ruff", "mypy"]
all  = ["simulacrum[ml,jax,meshing,dev,docs]"]
```

## Building the Rust Backend

```bash
cd src/simulacrum/_backends/rust
maturin develop --release
```

## Building the C++ Backend

```bash
cd src/simulacrum/_backends/cpp
mkdir build && cd build
cmake .. && make
```

## Roadmap

- [ ] FVM assembler (Godunov, Roe flux schemes)
- [ ] SPH assembler with neighbour search
- [ ] Transient solvers (implicit/explicit time integration)
- [ ] Parallel assembly with MPI (mpi4py)
- [ ] GPU acceleration (CuPy / JAX backend)
- [ ] Higher-order elements (Tri6, Quad9, Tet10, Hex27)
- [ ] Nonlinear material models (J2 plasticity, Neo-Hookean)
- [ ] Neural operator architectures (DeepONet, FNO)
- [ ] Diffusion models for Bayesian inference

## License

MIT
