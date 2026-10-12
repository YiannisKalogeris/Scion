# Scion

**S**cientific **C**omputing **I**nfrastructure for **O**perator **N**umerics.

A modular computational mechanics library with Python orchestration and optional Rust/C++/Warp acceleration. Designed for research in numerical methods for PDEs, scientific machine learning, and uncertainty quantification.

## Project Structure

```
Scion/
├── src/scion/              # The library (reusable across projects)
│   ├── core/                    #   Domain, Field, DOF, logging abstractions
│   ├── mesh/                    #   Mesh generation (built-in + gmsh), import via meshio
│   ├── discretization/          #   Numerical schemes
│   │   ├── fem/                 #     Finite Element Method (+ nonlinear assembly, quadrature state)
│   │   ├── mpm/                 #     Material Point Method (APIC/PIC/FLIP, GIMP, GPIC, Warp backend)
│   │   ├── fdm/                 #     Finite Difference Method
│   │   ├── fvm/                 #     Finite Volume Method (scalar + 1-D Euler)
│   │   └── sph/                 #     Smoothed Particle Hydrodynamics (scaffold)
│   ├── materials/               #   Constitutive models
│   │   ├── fracture/            #     Phase-field brittle fracture (composable)
│   │   ├── plasticity/          #     Rate-independent plasticity (composable)
│   │   ├── damage/              #     Continuum damage (local and gradient-regularised)
│   │   ├── micropolar*.py       #     Cosserat elasticity and elastoplasticity
│   │   ├── eshelby.py           #     Material-manifold quantities, J-integral
│   │   └── notation.py          #     Mandel / Voigt tensor conversions
│   ├── multiphysics/            #   Coupled two-field problems (staggered drivers)
│   ├── shakedown/               #   Direct methods (Melan, Linear Matching, Sloan elements)
│   ├── fatigue/                 #   Initiation life: counting, life curves, accumulation
│   ├── solvers/                 #   Linear, Newton-Raphson, staggered, bound-constrained
│   ├── boundary/                #   Dirichlet / Neumann BC handling
│   ├── geometry/                #   Shape generation, SDF rendering, harmonic maps
│   ├── ml/                      #   Machine learning (MLP, GNN, U-Net, training, surrogates)
│   ├── uq/                      #   Uncertainty quantification (sampling, MCMC, PCE, GP)
│   ├── optimization/            #   Optimisation (gradient, CMA-ES, Bayesian, OED)
│   ├── io/                      #   VTK/VTU/VTI/PVD output via meshio
│   ├── postprocessing/          #   PyVista-based visualisation
│   └── _backends/               #   Native acceleration
│       ├── rust/                #     PyO3/maturin Rust extension
│       └── cpp/                 #     pybind11/Eigen C++ extension
│
├── projects/                    # Research projects (consumers of the library)
│   ├── configurational_mpm/     #   Eshelby stress and pseudomomentum in MPM
│   ├── diffusion_preconditioner/#   Two-level preconditioning, with measured tables
│   └── nsf_domain_mapping/      #   Neural Spline Flow domain-to-square mapping
│
├── tests/                       # Unit and verification tests
├── examples/                    # Self-contained demos and benchmarks
│   ├── fem/                     #   Poisson, elasticity, pullback FEM
│   ├── mpm/                     #   Two-disk collision, block drop, GPIC bar
│   ├── fracture/                #   Single-edge notched tension
│   └── plasticity/              #   Thick-walled cylinder
├── benchmarks/                  # Performance benchmarks
├── docs/                        # Documentation source
├── pyproject.toml
└── README.md
```

## Design Principles

- **Python first**: All orchestration in Python. NumPy/SciPy for numerics.
- **Swappable discretizations**: FEM, MPM, FDM, FVM, SPH share the same `Domain → LinearSystem → Solver` pipeline.
- **Composition over inheritance**: Fracture and plasticity models are assembled from orthogonal, independently testable axes rather than one class per published model.
- **Constitutive models know nothing about meshes**: they operate on batched tensors at quadrature points, so a model written for FEM works unchanged in MPM or a meshfree scheme.
- **Verify against closed forms**: every derivative is checked against finite differences, and every physics module against an analytical solution or a published benchmark.
- **Library vs. projects**: Reusable components live in `src/scion/`; research workflows live in `projects/`.
- **Compiled backends optional**: Rust, C++ and NVIDIA Warp for hot loops. The library works without them.
- **VTK output**: Write `.vtu` / `.vti` / `.pvd` and visualise in ParaView or interactively with PyVista.

## Quick Start

```bash
pip install -e ".[dev]"          # core library
pip install -e ".[all]"          # everything

pytest tests/ -q                 # 1311 tests, ~2 minutes

python examples/fem/poisson_2d.py
python -m examples.mpm.two_disks
python -m examples.plasticity.thick_cylinder
```

## Subpackage Overview

### Core Simulation

| Subpackage | Purpose |
|---|---|
| `core` | Domain, Field, DOF management, logging |
| `mesh` | Mesh generation and import (structured, unstructured, gmsh) |
| `discretization.fem` | Linear and nonlinear assembly, persistent quadrature-point state |
| `discretization.mpm` | Material Point Method with NumPy and Warp backends, explicit / implicit / micropolar |
| `discretization.fdm` | Finite differences by Kronecker-sum assembly, verified by observed order |
| `discretization.fvm` | Scalar advection-diffusion and 1-D Euler (HLL, HLLC, exact Riemann) |
| `shakedown` | Melan, Linear Matching and Sloan equilibrium elements |
| `fatigue` | Rainflow counting, life curves, damage accumulation |
| `materials` | Elastic, thermal, hyperelastic, phase-field fracture, plasticity |
| `multiphysics` | Coupled two-field problems with staggered solution |
| `solvers` | Direct, Krylov, Newton-Raphson, staggered, bound-constrained |
| `boundary` | Dirichlet and Neumann boundary condition application |
| `io` | VTK/VTU/VTI/PVD output via meshio |
| `postprocessing` | Interactive 3D visualisation with PyVista |

### Material Point Method

| Module | Purpose |
|---|---|
| `mpm.basis` | Linear hat, quadratic/cubic B-spline, uGIMP with particle domains |
| `mpm.transfer` | PIC, FLIP and APIC particle–grid transfers |
| `mpm.stepper` | Explicit four-stage solver (stencil → P2G → grid → G2P) |
| `mpm.gpic` | Generalised Particle-in-Cell: total-Lagrangian FE bodies on an Eulerian grid |
| `mpm.warp` | NVIDIA Warp kernels; same physics on CPU or CUDA, ~30× over NumPy |

Particles are stored struct-of-arrays and the algorithms are written in "kernel shape" — explicit stencils, scatter/gather, closed-form small-matrix algebra — so the NumPy reference and the Warp kernels stay line-for-line comparable. A cross-check test pins the two backends together.

### Phase-Field Fracture

A model is composed from four orthogonal choices, each with its own registry:

```python
model = PhaseFieldModel(E, nu, Gc, ell,
                        split="spectral",        # none | amor | spectral
                        degradation="quadratic", # quadratic | cubic | rational
                        dissipation="AT2",       # AT1 | AT2
                        coupling="hybrid")       # isotropic | anisotropic | hybrid
```

| Axis | Module | Options |
|---|---|---|
| Energy split | `fracture.splits` | No split, volumetric/deviatoric (Amor), spectral (Miehe) |
| Degradation | `fracture.degradation` | Quadratic, cubic (Borden), rational (Lorentz/Wu) |
| Dissipation | `fracture.dissipation` | AT1 (elastic threshold, compact support), AT2 |
| Coupling | `fracture.coupling` | Isotropic, anisotropic, hybrid |

Adding a variant means one class and one registry entry. AT1 requires a bound-constrained phase-field solve, which the model declares through `needs_bound_constraint` so the driver dispatches automatically.

### Plasticity

Three orthogonal axes plus a general closest-point return map:

```python
model = PlasticityModel(E, nu,
                        criterion="j2",                      # or drucker-prager
                        isotropic=Voce(250.0, 400.0, 20.0),
                        flow=NonAssociative.drucker_prager(psi=10.0),
                        kinematic=LinearKinematic(2000.0))
```

| Axis | Module | Options |
|---|---|---|
| Yield criterion | `plasticity.yield_criteria` | von Mises (J2), Drucker–Prager (with Mohr–Coulomb calibrations) |
| Hardening | `plasticity.hardening` | Perfect, linear, Voce, Swift, Ludwik; Prager and Armstrong–Frederick kinematic |
| Flow rule | `plasticity.flow_rules` | Associative, non-associative (independent dilatancy angle) |

The return map solves for the *relative* stress, which eliminates the back stress and keeps kinematic hardening tractable. The **consistent tangent is obtained by reusing the converged Jacobian** rather than deriving a closed form per model, so it is exact for any combination of the three axes and for the algorithm as implemented — which is what gives quadratic Newton convergence.

Everything works on full 3-D tensors regardless of element dimension: in plane strain `σ₃₃` is nonzero and enters `J₂`, so reducing to in-plane components would give the wrong yield surface.

### Verification tools

| Module | Purpose |
|---|---|
| `solvers.tangent_check` | Finite-difference verification of any tangent operator |
| `solvers.staggered` | Alternating driver with a normalised-slope stopping criterion |
| `solvers.bound_constrained` | Primal-dual active set for `d ≥ 0` and irreversibility constraints |
| `fem.quadrature_state` | Persistent Gauss-point state with trial/committed semantics |

`check_tangent` takes callables, so the same test applies to plasticity, hyperelasticity and phase-field, at the material point or at the structural level. A wrong tangent still converges — linearly instead of quadratically — so nothing but an explicit check distinguishes the two.

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

## Verification

Physics modules are validated against closed-form solutions or published benchmarks, not only against themselves.

| Benchmark | Reference | Result |
|---|---|---|
| Thick-walled cylinder, perfect plasticity | `p_lim = 2Y ln(b/a)/√3` | 0.09% on the limit pressure; plastic front tracks the analytical `p(c)` within 4% |
| Single-edge notched tension | Ambati, Gerasimov & De Lorenzis (2015) | Peak load 0.645 kN vs 0.70; peak displacement matches |
| Two-disk collision (MPM) | Sulsky et al. | Momentum conserved to 1e-14 |
| GPIC bar in gravity | Nguyen et al. (2020) | 1.6% on the tip deflection against the analytical rod |
| 1-D crack profiles | `exp(-\|x\|/ℓ)` (AT2), `(1-\|x\|/2ℓ)²` (AT1) | First-order convergence; surface energy recovers `Gc × length` |
| Shakedown of a thick cylinder | Analytical collapse-governed limit | 99.7%, approached from below |
| Sod shock tube | Toro's exact star state | `p* = 0.30313`, `u* = 0.92745`, shock at `x = 0.850431` |
| FDM schemes | Theoretical order of accuracy | Observed order matches theory for every scheme |
| Rainflow counting | ASTM E1049-85 worked example | Exact |
| AMG-preconditioned CG | Mesh-independence on Poisson | 10/11/14/14 iterations over a 36× growth in unknowns |
| Curved cantilever (micropolar MPM) | Bathe & Bolourchi; Kozar & Ibrahimbegović | Within 2.6% of the analytical solution |
| Torsional rigidity (micropolar) | `K = GJ`, exact for a circular bar | 0.36% at zero internal length |

## Optional Dependencies

```toml
[project.optional-dependencies]
ml   = ["torch>=2.0"]
jax  = ["jax>=0.4", "jaxlib>=0.4", "equinox>=0.11", "optax>=0.2", "diffrax>=0.5"]
warp = ["warp-lang>=1.0"]
meshing = ["gmsh>=4.11"]
viz  = ["pyvista", "meshio"]
dev  = ["pytest>=7.0", "pytest-benchmark", "ruff", "mypy"]
all  = ["scion[ml,jax,warp,meshing,viz,dev,docs]"]
```

## Building the Accelerated Backends

```bash
# Rust
cd src/scion/_backends/rust && maturin develop --release

# C++
cd src/scion/_backends/cpp && mkdir build && cd build && cmake .. && make

# Warp needs no build step; GPU execution requires a CUDA device
pip install warp-lang
```

## Known Limitations

Stated explicitly rather than discovered at runtime — each of these raises with a clear message rather than silently approximating:

- **Plane stress plasticity** requires solving for `ε₃₃` inside the return map; only plane strain and 3-D are supported.
- **Mohr–Coulomb and Tresca** have surfaces with corners and need a multi-surface return map.
- **Armstrong–Frederick kinematic hardening** cannot be eliminated into the relative-stress form used by the return map.
- **Drucker–Prager with zero dilatancy** beyond the cone apex. The general apex return *is* implemented; this case stays refused for a structural reason — at zero dilatancy the flow rule is purely deviatoric, so nothing can carry the pressure back to the tip.
- **GPIC contact** is no-slip only, and the background grid must resolve the FE mesh.
- Phase-field fracture uses **displacement control**, which cannot follow a true snap-back.

## Roadmap

- [x] Material Point Method (explicit APIC, GIMP, GPIC) with GPU backend
- [x] Phase-field brittle fracture (AT1/AT2, three energy splits, three couplings)
- [x] J2 and Drucker–Prager plasticity with consistent tangents
- [x] Micropolar (Cosserat) elasticity, elastoplasticity and phase-field fracture
- [x] Shakedown and limit analysis; fatigue initiation life; continuum damage
- [x] Configurational mechanics: Eshelby stress, J-integral, explicit MPM cracks
- [x] Nonlinear material models (Neo-Hookean, hyperelastic)
- [ ] Plane-stress return mapping
- [ ] Multi-surface return map (Mohr–Coulomb, Tresca, tension cutoff)
- [ ] Arc-length control for snap-back
- [x] Implicit MPM (quasi-static Newton, load-stepped, verified by manufactured solution)
- [x] FVM assembler (upwind/central, MUSCL, HLL/HLLC/exact Riemann; Roe deliberately refused)
- [ ] SPH assembler with neighbour search
- [ ] Transient solvers (implicit/explicit time integration)
- [ ] Parallel assembly with MPI (mpi4py)
- [ ] Higher-order elements (Tri6, Quad9, Tet10, Hex27)
- [ ] Neural operator architectures (DeepONet, FNO)
- [ ] Diffusion models for Bayesian inference

## Citing

If you use this library, please cite the repository.

## License

Apache License 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE).
