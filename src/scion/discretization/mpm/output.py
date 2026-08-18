"""
MPM result output.

Kept separate from ``scion/io/``: the writers there are generic and know
nothing about particles or constitutive laws, while this module knows how to
turn a :class:`ParticleSet` plus a material into the derived fields worth
looking at. Same split as FEM assembly versus VTK output.

Typical use inside a run::

    from scion.discretization.mpm.output import ParticleOutput

    out = ParticleOutput("results/block", material=material)
    stepper.run(particles, n_steps=4000,
                callback=out, callback_every=50)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scion.core import small_linalg as sla
from scion.io.particles import write_particles_vtu
from scion.io.time_series import TimeSeriesWriter
from scion.io.uniform_grid import flatten_for_vtk, write_image_vti
from .particles import ParticleSet


# ---------------------------------------------------------------------------
# Derived per-particle fields
# ---------------------------------------------------------------------------

def cauchy_stress(particles: ParticleSet, material) -> tuple[np.ndarray, np.ndarray]:
    """Cauchy stress ``sigma = (1/J) P F^T`` and ``J``."""
    P, J = material.pk1(particles.F)
    sigma = np.einsum("pij,pkj->pik", P, particles.F) / J[:, None, None]
    return sigma, J


def von_mises(sigma: np.ndarray) -> np.ndarray:
    """Von Mises equivalent stress.

    For 2-D input the tensor is embedded in 3-D with a zero out-of-plane
    component. That is the plane-*stress* assumption; if your 2-D model is
    really plane strain, supply sigma_zz yourself before calling this.
    """
    n, d = sigma.shape[0], sigma.shape[1]
    if d == 2:
        s = np.zeros((n, 3, 3), dtype=sigma.dtype)
        s[:, :2, :2] = sigma
    else:
        s = sigma
    p = np.trace(s, axis1=1, axis2=2) / 3.0
    dev = s - p[:, None, None] * np.eye(3)
    return np.sqrt(1.5 * np.einsum("pij,pij->p", dev, dev))


def particle_fields(particles: ParticleSet, material=None) -> dict:
    """Assemble the point-data dictionary written for each frame."""
    J = particles.jacobian()
    fields = {
        "velocity": particles.v,
        "speed": np.linalg.norm(particles.v, axis=1),
        "J": J,
        "volumetric_strain": J - 1.0,
        "mass": particles.mass,
        "volume": J * particles.vol0,
    }
    if material is not None:
        sigma, _ = cauchy_stress(particles, material)
        d = sigma.shape[1]
        fields["pressure"] = -np.trace(sigma, axis1=1, axis2=2) / d
        fields["von_mises"] = von_mises(sigma)
        # separate scalar components rather than one 9-component tensor:
        # ParaView copes with tensors, but meshio (and therefore the Blender
        # addon) silently drops arrays with more than three components
        names = ("x", "y", "z")
        for i in range(d):
            for j in range(i, d):
                fields[f"sigma_{names[i]}{names[j]}"] = sigma[:, i, j]
    return fields


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class ParticleOutput:
    """Callable suitable for ``ExplicitMPMStepper.run(callback=...)``.

    Writes one ``.vtu`` per frame plus a ``.pvd`` collection carrying physical
    time. Optionally also dumps the background grid mass as ``.vti`` -- useful
    when debugging, since spurious mass on nodes far from the material is an
    immediate sign of a broken stencil.

    Parameters
    ----------
    directory   : output directory, created if absent
    material    : optional; enables the stress fields
    name        : basename of the series
    grid        : optional CartesianGrid; enables grid output
    verbose     : also print a one-line diagnostic per frame
    body_force  : same vector passed to the stepper; lets the momentum
                  diagnostic subtract the body-force impulse
    """

    def __init__(self, directory, material=None, name: str = "particles",
                 grid=None, verbose: bool = True, body_force=None):
        self.directory = Path(directory)
        self.body_force = (np.zeros(3) if body_force is None
                           else np.asarray(body_force, dtype=float))
        self.material = material
        self.grid = grid
        self.verbose = verbose
        self.series = TimeSeriesWriter(self.directory, name, suffix=".vtu")
        self.grid_series = (
            TimeSeriesWriter(self.directory, f"{name}_grid", suffix=".vti")
            if grid is not None else None
        )
        self._p0 = None
        self._m_total = 0.0

    # ------------------------------------------------------------------
    def __call__(self, step: int, t: float, particles: ParticleSet) -> None:
        fields = particle_fields(particles, self.material)
        self.series.add(t, write_particles_vtu, particles.x, point_data=fields)

        if self.grid_series is not None:
            self._write_grid(t, particles)

        if self.verbose:
            if self._p0 is None:
                self._p0 = particles.linear_momentum()
                self._m_total = particles.mass.sum()
            # momentum is conserved only up to the impulse of the body force,
            # so compare against p0 + m g t rather than p0
            expected = self._p0 + self._m_total * self.body_force * t
            dp = np.linalg.norm(particles.linear_momentum() - expected)
            print(f"  frame {len(self.series)-1:4d}  t={t:8.4f}  "
                  f"KE={particles.kinetic_energy():.4e}  "
                  f"minJ={particles.jacobian().min():.4f}  "
                  f"|dp|={dp:.2e}")

    # ------------------------------------------------------------------
    def _write_grid(self, t: float, particles: ParticleSet) -> None:
        grid = self.grid
        st = grid.build_stencil(particles.x)
        mass = grid.zeros_scalar()
        np.add.at(mass, st.node_id, st.w * particles.mass[:, None])

        # VTK ImageData expects x fastest; the grid stores the last axis fastest
        mass = flatten_for_vtk(mass.reshape(grid.node_shape))

        origin = grid.origin - grid.pad * grid.dx
        self.grid_series.add(
            t, write_image_vti, origin, grid.dx, grid.node_shape,
            point_data={"mass": mass},
        )
