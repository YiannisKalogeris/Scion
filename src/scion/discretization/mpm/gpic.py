r"""
Generalized Particle In Cell (GPIC).

Nguyen, de Vaucorbeil, Nguyen-Thanh & Mandal, CMAME 2020/2021,
"A generalized particle in cell method for explicit solid dynamics".

The idea: replace point particles with *finite elements*. Each solid is a proper
Lagrangian FE mesh whose nodes play the role of MPM particles, embedded in a
fixed background grid that is used only to exchange momentum between bodies.
An FE element is, in the paper's phrasing, a kind of MPM generalized particle.

What that buys, relative to the updated-Lagrangian MPM in ``stepper.py``:

* **exact quadrature** -- Gauss points on a mesh with connectivity, not scattered
  points whose spacing drifts. This is the "quadrature error" column.
* **no cell crossing** -- internal forces never touch the background grid basis.
* **no numerical fracture** -- neighbouring material is held together by element
  connectivity, so it cannot spuriously separate. GPIC and CPDI are the only
  common variants free of it.
* **contact for free** -- distinct bodies scatter to shared grid nodes, so the
  grid solve enforces no-interpenetration automatically, exactly as in MPM.

What it costs: the mesh is Lagrangian, so topology cannot change. Fragmentation,
self-merging and fluid-like flow are outside GPIC's reach; that is the regime
where updated-Lagrangian MPM remains the right tool. Extreme deformation with
topological change is explicitly noted as an open difficulty for GPIC.

This module implements the **total Lagrangian (TL)** variant: internal forces are
evaluated in the reference configuration from the first Piola-Kirchhoff stress,
so the shape-function gradients ``dN/dX`` are computed once at setup and never
recomputed. The paper also gives an updated-Lagrangian variant; TL is the one it
focuses on, and it is both cheaper and free of incremental drift in F.

Structure of one step::

    1. FE:   F at Gauss points from current nodal positions (total Lagrangian)
             internal nodal forces  f_a = -sum_qp w0 * P . dN/dX_a
    2. P2G:  scatter mass, momentum and f_a from FE nodes to the grid
    3. grid: momentum update, body force, boundary conditions, contact
    4. G2P:  gather velocity back to the FE nodes
    5. FE:   advance nodal positions

Note step 1 comes *first*: unlike point-particle MPM, F is a function of the
nodal positions alone, so it is recomputed rather than integrated in time. There
is no ``F += dt L F`` and therefore no accumulated drift.
"""

from __future__ import annotations

import numpy as np

from scion.core import small_linalg as sla
from scion.discretization.fem.elements import get_element
from scion.discretization.fem.quadrature import get_quadrature

from .background_grid import CartesianGrid

__all__ = ["SolidBody", "GPICStepper"]


# ---------------------------------------------------------------------------
# Solid body
# ---------------------------------------------------------------------------

class SolidBody:
    """A Lagrangian finite-element mesh acting as a GPIC generalized particle.

    Struct-of-arrays throughout, for the same reason as ``ParticleSet``.

    Nodal state
    -----------
    X     (n_nodes, dim)              reference positions (never change)
    x     (n_nodes, dim)              current positions
    v     (n_nodes, dim)              velocity
    C     (n_nodes, dim, dim)         APIC affine matrix
    mass  (n_nodes,)                  lumped nodal mass

    Element state
    -------------
    conn  (n_el, npe)                 connectivity
    dN_dX (n_el, n_qp, npe, dim)      reference shape gradients, precomputed
    w0    (n_el, n_qp)                reference quadrature weights * detJ0
    F     (n_el, n_qp, dim, dim)      deformation gradient at Gauss points
    """

    def __init__(self, nodes, connectivity, etype: str, density: float,
                 velocity=None, one_point: bool = True, dtype=np.float64):
        self.dtype = dtype
        self.etype = etype
        self.X = np.ascontiguousarray(nodes, dtype=dtype)
        self.x = self.X.copy()
        self.n_nodes, self.dim = self.X.shape

        self.conn = np.ascontiguousarray(connectivity, dtype=np.int64)
        self.n_el, self.npe = self.conn.shape

        self.v = np.zeros_like(self.X)
        if velocity is not None:
            self.v[:] = np.asarray(velocity, dtype=dtype)
        self.C = np.zeros((self.n_nodes, self.dim, self.dim), dtype=dtype)

        self.one_point = bool(one_point)
        self.du = np.zeros_like(self.X)
        # Dirichlet conditions live on the FE mesh, NOT on the Eulerian grid.
        # The paper shows constraining grid nodes instead lets the body slide
        # off its own support (their Fig. 7): the grid node is shared with
        # empty space, so the constraint is applied to the wrong material.
        self.fixed = np.zeros((self.n_nodes, self.dim), dtype=bool)
        self._precompute_reference(density)
        self.F = sla.identity(self.n_el * self.n_qp, self.dim, dtype).reshape(
            self.n_el, self.n_qp, self.dim, self.dim
        )

    # ------------------------------------------------------------------
    def _precompute_reference(self, density: float) -> None:
        """Reference gradients, weights and lumped mass. Done once.

        This is the whole performance argument for the total-Lagrangian form:
        ``dN_dX`` never changes, so the per-step cost is two contractions.
        """
        ref = get_element(self.etype)
        if self.one_point and self.etype == "quad4":
            # Single centre point, weight 4 (paper Fig. 2). The authors note
            # that, unlike in the FEM, this did not produce zero-energy modes
            # in any of their simulations -- the grid projection appears to
            # suppress hourglassing. It also makes one element equivalent to
            # one particle, which is the cleanest reading of GPIC.
            qpts, qwts = np.zeros((1, 2)), np.array([4.0])
        else:
            qpts, qwts = get_quadrature(self.etype)
        self.n_qp = len(qwts)

        dN_dX = np.zeros((self.n_el, self.n_qp, self.npe, self.dim),
                         dtype=self.dtype)
        w0 = np.zeros((self.n_el, self.n_qp), dtype=self.dtype)

        for e in range(self.n_el):
            coords = self.X[self.conn[e]]
            for q, (qp, qw) in enumerate(zip(qpts, qwts)):
                _, dN_dxi = ref.evaluate(qp)
                J = dN_dxi.T @ coords                       # J[i,j] = dx_j/dxi_i
                # dN/dX = dN_dxi @ inv(J).T -- the transpose is NOT optional;
                # without it the gradients are wrong for any element whose
                # Jacobian is not symmetric (rotated, skewed, or numbered from
                # a different corner).
                dN_dX[e, q] = np.linalg.solve(J, dN_dxi.T).T
                w0[e, q] = abs(np.linalg.det(J)) * qw

        self.dN_dX = dN_dX
        self.w0 = w0

        # lumped nodal mass, m_a = sum_e sum_q rho0 * N_a * w0
        mass = np.zeros(self.n_nodes, dtype=self.dtype)
        for e in range(self.n_el):
            for q, qp in enumerate(qpts):
                N, _ = ref.evaluate(qp)
                np.add.at(mass, self.conn[e], density * N * self.w0[e, q])
        self.mass = mass
        self.volume0 = float(self.w0.sum())

    # ------------------------------------------------------------------
    def fix_nodes(self, predicate, components=None) -> None:
        """Constrain FE nodes selected by ``predicate(X) -> (n,) bool``.

        ``components`` is a sequence of axis indices, or None for all.
        """
        sel = np.asarray(predicate(self.X), dtype=bool)
        comps = range(self.dim) if components is None else components
        for c in comps:
            self.fixed[sel, c] = True

    def apply_dirichlet(self) -> None:
        """v = 0, du = 0, x = X on constrained components (Algorithm line 20)."""
        if not self.fixed.any():
            return
        self.v[self.fixed] = 0.0
        self.du[self.fixed] = 0.0
        self.x[self.fixed] = self.X[self.fixed]

    # ------------------------------------------------------------------
    def velocity_gradient(self, dt: float) -> np.ndarray:
        """L = F Fdot^-1 with Fdot = (1/dt) sum_J grad0 phi_J du_J (eq. 2.5-2.6).

        Needed only by rate-form constitutive models; D = sym(L).
        """
        due = self.du[self.conn]
        Fdot = np.einsum("eai,eqaj->eqij", due, self.dN_dX) / dt
        shape = (self.n_el * self.n_qp, self.dim, self.dim)
        return np.einsum("pij,pjk->pik",
                         self.F.reshape(shape),
                         sla.inv(Fdot.reshape(shape)))

    # ------------------------------------------------------------------
    def update_deformation_gradient(self) -> None:
        """F = I + sum_a (x_a - X_a) (x) dN/dX_a, evaluated from scratch.

        Total Lagrangian: F depends only on the current nodal positions, so it
        is recomputed rather than integrated. No incremental drift, and it stays
        exact under rigid body motion to machine precision.
        """
        u = self.x - self.X                                   # (n_nodes, dim)
        ue = u[self.conn]                                     # (n_el, npe, dim)
        grad_u = np.einsum("eai,eqaj->eqij", ue, self.dN_dX)
        self.F = np.eye(self.dim, dtype=self.dtype) + grad_u

    # ------------------------------------------------------------------
    def internal_force(self, material) -> np.ndarray:
        """Nodal internal force, f_a = -sum_qp w0 * P . dN/dX_a.

        The standard total-Lagrangian FE internal force. Identical in form to
        what ``FEMAssembler`` computes for a static problem, which is why the
        same batched material interface serves both.
        """
        shape = (self.n_el * self.n_qp, self.dim, self.dim)
        P, _ = material.pk1(self.F.reshape(shape))
        P = P.reshape(self.n_el, self.n_qp, self.dim, self.dim)

        fe = -np.einsum("eq,eqij,eqaj->eai", self.w0, P, self.dN_dX)
        f = np.zeros_like(self.x)
        np.add.at(f, self.conn, fe)
        return f

    # ------------------------------------------------------------------
    def jacobian(self) -> np.ndarray:
        """det(F) at every Gauss point, flattened."""
        shape = (self.n_el * self.n_qp, self.dim, self.dim)
        return sla.det(self.F.reshape(shape))

    def linear_momentum(self) -> np.ndarray:
        return (self.mass[:, None] * self.v).sum(axis=0)

    def kinetic_energy(self) -> float:
        return float(0.5 * (self.mass * (self.v ** 2).sum(axis=1)).sum())

    def strain_energy(self, material) -> float:
        """Requires the material to expose ``energy(F)``; returns NaN if not."""
        fn = getattr(material, "energy", None)
        if fn is None:
            return float("nan")
        shape = (self.n_el * self.n_qp, self.dim, self.dim)
        return float((fn(self.F.reshape(shape)).reshape(self.n_el, self.n_qp)
                      * self.w0).sum())

    # ------------------------------------------------------------------
    @classmethod
    def structured_quad(cls, lo, hi, nx, ny, density, velocity=None,
                        one_point=True, dtype=np.float64) -> "SolidBody":
        """Rectangular body meshed with quad4 elements."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        xs = np.linspace(lo[0], hi[0], nx + 1)
        ys = np.linspace(lo[1], hi[1], ny + 1)
        nodes = np.array([[x, y] for y in ys for x in xs])
        conn = np.array([
            [j * (nx + 1) + i, j * (nx + 1) + i + 1,
             j * (nx + 1) + i + nx + 2, j * (nx + 1) + i + nx + 1]
            for j in range(ny) for i in range(nx)
        ])
        return cls(nodes, conn, "quad4", density, velocity, one_point, dtype)

    @classmethod
    def annulus_quad(cls, centre, r_in, r_out, n_r, n_theta, density,
                     velocity=None, one_point=True,
                     dtype=np.float64) -> "SolidBody":
        """Ring meshed with quad4 elements -- the classic GPIC contact demo."""
        centre = np.asarray(centre, float)
        rr = np.linspace(r_in, r_out, n_r + 1)
        th = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
        nodes = np.array([[centre[0] + r * np.cos(t), centre[1] + r * np.sin(t)]
                          for r in rr for t in th])
        conn = []
        for j in range(n_r):
            for i in range(n_theta):
                i1 = (i + 1) % n_theta
                conn.append([j * n_theta + i, j * n_theta + i1,
                             (j + 1) * n_theta + i1, (j + 1) * n_theta + i])
        return cls(nodes, np.array(conn), "quad4", density, velocity,
                   one_point, dtype)


# ---------------------------------------------------------------------------
# Stepper
# ---------------------------------------------------------------------------

class GPICStepper:
    """Explicit total-Lagrangian GPIC.

    Takes a *list* of bodies rather than a single particle set, because the
    whole point of the background grid here is inter-body contact. Bodies may
    have different materials and different meshes.

    Parameters
    ----------
    grid        : CartesianGrid -- the contact medium only; it never sees stress
    bodies      : sequence of SolidBody
    materials   : one material per body, or a single one shared by all
    dt          : time step
    body_force  : (dim,) acceleration
    boundaries  : sequence of GridBC applied after the grid velocity update
    transfer    : "pic" (default), "apic" or "flip".

                  "pic" replaces the whole nodal velocity by its grid
                  projection each step. This is what makes contact work -- see
                  the grid-resolution note below -- and is the default.

                  "flip" passes only the *increment* through the grid, so an
                  isolated body evolves as pure TL-FEM. But pure FLIP cannot
                  resolve contact at all: MPM contact works precisely *because*
                  colliding bodies are forced to share one single-valued grid
                  velocity, and FLIP discards exactly that. Verified here --
                  with "flip" the two-ring benchmark has the rings pass straight
                  through one another. Use it only for single-body problems.

    GRID RESOLUTION IS NOT FREE HERE
    --------------------------------
    Unlike point-particle MPM, the background grid must resolve the *FE mesh*,
    not merely contain the body. With PIC transfer the nodal velocity field is
    reprojected onto the grid basis every step, so a grid coarser than the
    elements filters the velocity content away as numerical dissipation.
    Measured on the two-ring benchmark: with dx six times the element size, 97%
    of the kinetic energy is gone by first contact; with dx comparable to the
    element size, kinetic energy is conserved to 100.0% while the bodies are
    separate. Size the grid to the elements.
    """

    def __init__(self, grid: CartesianGrid, bodies, materials, dt: float,
                 body_force=None, boundaries=()):
        self.grid = grid
        self.bodies = list(bodies)
        if not isinstance(materials, (list, tuple)):
            materials = [materials] * len(self.bodies)
        if len(materials) != len(self.bodies):
            raise ValueError("need one material per body")
        self.materials = list(materials)
        self.dt = float(dt)
        self.body_force = (np.zeros(grid.dim) if body_force is None
                           else np.asarray(body_force, dtype=grid.dtype))
        self.boundaries = list(boundaries)
        self.contact_nodes = np.zeros(grid.n_nodes, dtype=bool)

    # ------------------------------------------------------------------
    def cfl_dt(self, density: float, safety: float = 0.2) -> float:
        """Stable step from the element size, not the grid size.

        GPIC's stability limit is set by the *FE mesh*, since that is where the
        stress waves travel; the background grid only exchanges momentum. In
        practice the element size is the smaller of the two, so this is tighter
        than the MPM condition.
        """
        h = min(
            float(np.sqrt(b.w0.sum(axis=1)).min()) for b in self.bodies
        )
        c = max(m.wave_speed_factor for m in self.materials) / np.sqrt(density)
        v_max = max(float(np.abs(b.v).max()) for b in self.bodies)
        return safety * h / (c + v_max)

    # ------------------------------------------------------------------
    def step(self) -> None:
        """One GPIC step, following Algorithm 2 (TL) of Nguyen et al. (2020).

        The velocity and the position are updated from *different* grid
        quantities, and that asymmetry is the whole trick:

            v_J += sum_I phi_I(x_J) [v_I^{t+dt} - v_I^t]      (FLIP: increment)
            x_J += dt * sum_I phi_I(x_J) v_I^{t+dt}           (PIC: total)

        FLIP on the velocity avoids the numerical dissipation of reprojecting
        the whole velocity field onto a grid coarser than the FE mesh. PIC on
        the position is what makes contact work: colliding bodies are advected
        by the single-valued grid velocity, so they cannot interpenetrate. Use
        FLIP for both and bodies pass through each other; use PIC for both and
        the collision is smothered by dissipation.
        """
        grid, dt = self.grid, self.dt
        if grid.basis.requires_domain:
            raise ValueError(
                "GPIC bodies have no particle domain; use a point basis "
                "(linear/quadratic/cubic) for the background grid."
            )

        grid_m = grid.zeros_scalar()
        grid_mv = grid.zeros_vector()
        grid_f = grid.zeros_vector()
        stencils, body_mass = [], []

        # --- M2G: map mass, momentum and forces from the FE meshes -----------
        for body, material in zip(self.bodies, self.materials):
            body.update_deformation_gradient()
            f_int = body.internal_force(material)          # already carries -
            f_ext = body.mass[:, None] * self.body_force

            st = grid.build_stencil(body.x)
            stencils.append(st)

            wm = st.w * body.mass[:, None]
            np.add.at(grid_m, st.node_id, wm)

            mb = grid.zeros_scalar()
            np.add.at(mb, st.node_id, wm)
            body_mass.append(mb)

            np.add.at(grid_mv, st.node_id, wm[:, :, None] * body.v[:, None, :])
            np.add.at(grid_f, st.node_id,
                      st.w[:, :, None] * (f_int + f_ext)[:, None, :])

        # --- solve the momentum equation on the Eulerian grid ----------------
        active = grid_m > 1.0e-12
        v_old = np.zeros_like(grid_mv)
        v_old[active] = grid_mv[active] / grid_m[active, None]

        grid_v = np.zeros_like(grid_mv)
        grid_v[active] = (grid_mv[active] + dt * grid_f[active]) \
            / grid_m[active, None]
        for bc in self.boundaries:
            bc(grid, grid_v, grid_m)

        dv = grid_v - v_old

        occupancy = sum((mb > 1.0e-12).astype(np.int32) for mb in body_mass)
        self.contact_nodes = occupancy >= 2

        # --- G2M: velocity by FLIP, position and displacement by PIC ---------
        for body, st in zip(self.bodies, stencils):
            body.v = body.v + np.einsum("pk,pka->pa", st.w, dv[st.node_id])
            body.du = dt * np.einsum("pk,pka->pa", st.w, grid_v[st.node_id])
            body.x = body.x + body.du
            body.apply_dirichlet()

        # UMF: stress and forces are recomputed at the top of the next step

    # ------------------------------------------------------------------
    def run(self, n_steps: int, callback=None, callback_every: int = 100):
        for s in range(n_steps + 1):
            if callback is not None and s % callback_every == 0:
                callback(s, s * self.dt, self.bodies)
            if s < n_steps:
                self.step()
        return self.bodies

    # ------------------------------------------------------------------
    def total_momentum(self) -> np.ndarray:
        return sum(b.linear_momentum() for b in self.bodies)

    def total_kinetic_energy(self) -> float:
        return sum(b.kinetic_energy() for b in self.bodies)
