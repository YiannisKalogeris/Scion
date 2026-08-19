"""
Example: Solve the 2-D Poisson equation on a unit square.

    -Δu = 1   on Ω = (0,1)²
     u  = 0   on ∂Ω

Uses Quad4 elements with the FEM assembler and a direct solver.
"""

from pathlib import Path

import numpy as np

from scion.core import Domain
from scion.mesh import rectangle_mesh
from scion.materials import IsotropicThermal
from scion.discretization.fem import FEMAssembler
from scion.boundary import DirichletBC, apply_dirichlet_elimination
from scion.solvers import DirectSolver
from scion.io import write_vtk
from scion.postprocessing import plot_field

RESULTS = Path(__file__).resolve().parent / "results"


def main():
    # 1. Create mesh
    mesh = rectangle_mesh((0, 1), (0, 1), nx=20, ny=20, etype="quad4")
    print(f"Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")

    # 2. Set up domain
    domain = Domain(mesh=mesh, dim=2)
    domain.add_material("default", IsotropicThermal(k=1.0))

    # 3. Assemble
    assembler = FEMAssembler(domain, n_components=1)
    system = assembler.assemble()
    K, f = system.matrix, system.rhs

    # 4. Apply source term (f = 1 everywhere — lumped)
    f[:] = 1.0 * (1.0 / mesh.n_nodes)  # simplified uniform load

    # 5. Apply Dirichlet BCs on all boundaries
    bcs = [
        DirichletBC(mesh.boundary_nodes("left"), component=0, value=0.0),
        DirichletBC(mesh.boundary_nodes("right"), component=0, value=0.0),
        DirichletBC(mesh.boundary_nodes("bottom"), component=0, value=0.0),
        DirichletBC(mesh.boundary_nodes("top"), component=0, value=0.0),
    ]
    K, f = apply_dirichlet_elimination(K, f, assembler.dof_map, bcs)

    # 6. Solve
    solver = DirectSolver()
    u = solver.solve(K, f)
    print(f"Solution range: [{u.min():.6f}, {u.max():.6f}]")

    # 7. Write output
    #    Anchored to this file, not the working directory: a bare relative path
    #    here dropped the .vtu wherever the example happened to be launched
    #    from, which for `python -m examples.fem.poisson_2d` is the repo root.
    out = RESULTS / "poisson_result.vtu"
    write_vtk(out, mesh, point_data={"temperature": u})
    print(f"Wrote {out}")

    # 8. (Optional) interactive plot
    # plot_field(mesh, u, field_name="temperature")


if __name__ == "__main__":
    main()
