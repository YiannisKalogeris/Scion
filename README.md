# Simulacrum

A modular computational mechanics library with Python orchestration and optional Rust/C++ acceleration.

## Architecture

```
simulacrum/
├── core/                # Domain, Field, DOF abstractions
├── mesh/                # Mesh generation (built-in + gmsh), import via meshio
├── discretization/      # Numerical schemes
│   ├── fem/             #   Finite Element Method (implemented)
│   ├── fdm/             #   Finite Difference Method (implemented)
│   ├── fvm/             #   Finite Volume Method (scaffold)
│   └── sph/             #   Smoothed Particle Hydrodynamics (scaffold)
├── materials/           # Constitutive models (linear elastic, thermal, hyperelastic)
├── solvers/             # Linear (direct, Krylov) and nonlinear (Newton-Raphson) solvers
├── boundary/            # Dirichlet / Neumann BC handling
├── io/                  # VTK/VTU output via meshio
├── postprocessing/      # PyVista-based visualisation
└── _backends/           # Native acceleration
    ├── rust/            #   PyO3/maturin Rust extension
    └── cpp/             #   pybind11/Eigen C++ extension
```

## Design Principles

- **Python first**: All orchestration in Python. NumPy/SciPy for numerics.
- **Swappable discretizations**: FEM, FDM, FVM, SPH share the same `Domain → LinearSystem → Solver` pipeline.
- **Compiled backends optional**: Rust (via maturin/PyO3) and C++ (via pybind11/Eigen) for hot loops. The library works without them.
- **Meshing is external**: Use gmsh or meshio for mesh generation and import.
- **VTK output**: Write `.vtu` files and visualise in ParaView or interactively with PyVista.

## Quick Start

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run an example
python examples/fem/poisson_2d.py
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

## Dependencies

| Package  | Purpose                     |
|----------|-----------------------------|
| numpy    | Array operations            |
| scipy    | Sparse matrices, solvers    |
| meshio   | Mesh I/O (VTK, gmsh, etc.) |
| gmsh     | Mesh generation             |
| pyvista  | Interactive visualisation   |
| vtk      | VTK data structures         |

## Roadmap

- [ ] FVM assembler (Godunov, Roe flux schemes)
- [ ] SPH assembler with neighbour search
- [ ] Transient solvers (implicit/explicit time integration)
- [ ] Parallel assembly with MPI (mpi4py)
- [ ] GPU acceleration (CuPy / JAX backend)
- [ ] Higher-order elements (Tri6, Quad9, Tet10, Hex27)
- [ ] Nonlinear material models (J2 plasticity, Neo-Hookean)
