"""Explicit crack surfaces for MPM, and configurational-force-driven growth.

The obstacle this exists to remove
----------------------------------
Standard MPM carries a **single-valued** field on the background grid, so
material on the two sides of a would-be crack inside one cell is forced to move
together: a displacement discontinuity finer than a cell simply cannot be
represented. That is not a shortcoming of the constitutive model or of the
solver, and no amount of mesh refinement fixes it -- the grid is what couples
the two faces.

The fix here is **node splitting by visibility**: a grid node that a particle
cannot "see", because the straight line between them crosses the crack, is
replaced for that particle by a *duplicate* node carrying its own degrees of
freedom. Material either side of the crack then has independent kinematics
while sharing the same grid. This is the static counterpart of the multiple
velocity fields of Nairn's CRAMP.

Why the remap is the whole implementation
-----------------------------------------
Every consumer of a stencil -- internal force, tangent assembly, the transfers
-- addresses nodes through ``stencil.node_id`` and accumulates with
``np.add.at``. Rewriting that array therefore splits the node everywhere at
once, with no change to any assembly loop. The cost is that the node count
grows per load step, so anything sized ``grid.n_nodes`` has to be sized by the
*effective* count instead.

Known limitation: plain visibility
----------------------------------
The visibility criterion makes the shape functions discontinuous along the ray
running ahead of the tip, because a node just beyond the tip is visible from
one side and hidden from the other with nothing in between. The standard
remedies are the *diffraction* and *transparency* methods, which taper the
blocking near the tip. Neither is implemented; for a tip well inside the body
and a contour taken a few cells out, the artefact sits inside the ring rather
than on it. This is a real approximation and not a detail -- it is the first
thing to suspect if the driving force depends on where the tip sits within a
cell.
"""

from __future__ import annotations

import numpy as np

from ...materials.eshelby import material_force

__all__ = ["CrackPath", "split_nodes", "crack_tip_force",
           "AnisotropicToughness", "propagation_direction"]


def _crosses(p1, p2, q1, q2):
    """``(n, k, m)`` -- does segment ``p1->p2`` properly cross ``q1->q2``?

    Orientation form rather than a parametric solve: it needs no division, so
    parallel segments do not have to be special-cased, they simply fail the
    sign tests. Strict inequalities throughout, so a line of sight that merely
    *touches* a crack vertex or runs along a face is not counted as blocked --
    the degenerate cases are measure-zero in position and treating them as
    unblocked keeps a particle sitting exactly on the crack line attached to
    its original node rather than flipping on round-off.
    """
    def cross(ax, ay, bx, by):
        return ax * by - ay * bx

    # p1, p2: (n, k, 2)   q1, q2: (m, 2)
    p1 = p1[..., None, :]
    p2 = p2[..., None, :]
    d_p = p2 - p1                      # (n, k, 1, 2)
    d_q = (q2 - q1)                    # (m, 2)

    r1 = p1 - q1                       # (n, k, m, 2)
    r2 = p2 - q1
    d1 = cross(d_q[..., 0], d_q[..., 1], r1[..., 0], r1[..., 1])
    d2 = cross(d_q[..., 0], d_q[..., 1], r2[..., 0], r2[..., 1])
    s1 = q1 - p1                       # (n, k, m, 2)
    s2 = q2 - p1
    d3 = cross(d_p[..., 0], d_p[..., 1], s1[..., 0], s1[..., 1])
    d4 = cross(d_p[..., 0], d_p[..., 1], s2[..., 0], s2[..., 1])
    return ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))


class CrackPath:
    """A polyline crack, 2-D, held in the **current** configuration.

    ``vertices`` runs from the mouth to the tip, so ``vertices[-1]`` is where
    growth happens and ``vertices[0]`` normally sits on a free surface.

    Held in the current configuration rather than the reference one because the
    grid it cuts is the current grid. That makes ``advect`` necessary: a crack
    is a *material* surface, so its vertices must ride with the material or the
    body will slide through a crack pinned in space. At small strain the
    difference is negligible, which is exactly why it is easy to omit and get
    away with until it matters.
    """

    def __init__(self, vertices):
        v = np.asarray(vertices, dtype=float)
        if v.ndim != 2 or v.shape[1] != 2:
            raise ValueError(f"crack vertices must be (m, 2), got {v.shape}")
        if len(v) < 2:
            raise ValueError("a crack needs at least two vertices")
        self.vertices = v

    def __repr__(self) -> str:
        return (f"CrackPath({len(self.vertices)} vertices, "
                f"tip={np.array2string(self.tip, precision=4)}, "
                f"length={self.length:.4g})")

    @property
    def tip(self) -> np.ndarray:
        return self.vertices[-1]

    @property
    def mouth(self) -> np.ndarray:
        return self.vertices[0]

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.vertices, axis=0),
                                    axis=1).sum())

    @property
    def tip_direction(self) -> np.ndarray:
        """Unit vector along the last segment, pointing towards the tip."""
        d = self.vertices[-1] - self.vertices[-2]
        return d / np.linalg.norm(d)

    def segments(self):
        """``(q1, q2)``, each ``(m, 2)``."""
        return self.vertices[:-1], self.vertices[1:]

    def blocked(self, particle_xy, node_xy, node_id) -> np.ndarray:
        """``(n, k)`` -- is the line of sight from particle to node cut?

        ``node_id`` is the stencil's ``(n, k)`` index array. Coordinates may be
        ``(., 2)`` or ``(., 3)``; a third component is **projected away**.

        That projection is exact rather than approximate for the case it is
        meant for. A plane-strain body is one or a few cells thick with the
        crack a *prism* parallel to z, so a line of sight crosses the crack
        surface if and only if its xy-projection crosses the polyline. It would
        not be exact for a crack inclined to z, which is a genuinely 3-D crack
        -- a surface rather than a curve -- and outside what this class
        represents.
        """
        particle_xy = np.asarray(particle_xy, dtype=float)[:, :2]
        node_xy = np.asarray(node_xy, dtype=float)[:, :2]
        p1 = np.broadcast_to(particle_xy[:, None, :], node_id.shape + (2,))
        p2 = node_xy[node_id]
        q1, q2 = self.segments()
        return _crosses(p1, p2, q1, q2).any(axis=-1)

    def extend(self, direction, length: float) -> "CrackPath":
        """A new path with the tip advanced. Does not mutate."""
        d = np.asarray(direction, dtype=float)
        n = np.linalg.norm(d)
        if n == 0.0:
            raise ValueError("cannot extend a crack along a zero direction")
        return CrackPath(np.vstack([self.vertices,
                                    self.tip + length * d / n]))

    def advect(self, displacement) -> "CrackPath":
        """A new path with every vertex moved by ``displacement`` ``(m, 2)``."""
        return CrackPath(self.vertices + np.asarray(displacement, dtype=float))


def split_nodes(node_id: np.ndarray, blocked: np.ndarray, n_nodes: int):
    """Remap blocked ``(particle, node)`` pairs onto duplicate nodes.

    Returns ``(new_node_id, n_total, origin)``. ``origin[j]`` is the grid node
    that duplicate ``n_nodes + j`` was made from, which is what lets a duplicate
    inherit its parent's Dirichlet constraint and its coordinates.

    One duplicate per *node*, not per particle: for a single crack the blocked
    particles of a given node all lie on the same side of it, so one extra copy
    is exactly enough. A branching or intersecting crack would need more, and
    this returns a two-field split rather than pretending otherwise.
    """
    node_id = np.asarray(node_id)
    blocked = np.asarray(blocked, dtype=bool)
    if blocked.shape != node_id.shape:
        raise ValueError(f"blocked {blocked.shape} does not match node_id "
                         f"{node_id.shape}")
    if not blocked.any():
        return node_id.copy(), n_nodes, np.zeros(0, dtype=np.int64)

    origin = np.unique(node_id[blocked])
    lookup = np.full(n_nodes, -1, dtype=np.int64)
    lookup[origin] = n_nodes + np.arange(len(origin))
    new_id = node_id.copy()
    new_id[blocked] = lookup[node_id[blocked]]
    return new_id, n_nodes + len(origin), origin


def crack_tip_force(sigma, volume, grad_w, node_id, n_nodes, node_xy,
                    tip, radius) -> np.ndarray:
    """Configurational force on the tip, by the domain form.

    Denzer's Eq. (27): the resultant of the nodal material forces over a domain
    surrounding the tip, rather than the single crack-tip node of Miehe's
    Eq. (26). The domain form is the accurate one and is what this returns.

    ``grad_w`` must be the **material** gradients. Passing spatial ones is wrong
    in meaning but right in shape and units, and at small strain the two agree
    closely enough to look plausible -- the same trap ``material_force``
    records.

    ``radius`` selects the domain. Read it as a ring test: the answer should be
    insensitive to it once the domain clears the tip's own cell, and a value
    that keeps drifting with ``radius`` means the field is not equilibrated or
    the contour is cutting the visibility artefact ahead of the tip.
    """
    contrib = material_force(sigma, volume, grad_w)
    dim = contrib.shape[-1]
    G = np.zeros((n_nodes, dim))
    np.add.at(G, node_id, contrib)
    inside = np.linalg.norm(node_xy[:len(G)] - np.asarray(tip), axis=1) <= radius
    # The minus is Denzer's, and it is not cosmetic: ``material_force`` returns
    # ``+V Sigma Grad w``, so without it the vector points *backwards* along
    # the crack and a growth criterion built on it would drive the tip the
    # wrong way while looking perfectly path-independent.
    return -G[inside].sum(axis=0)


class AnisotropicToughness:
    r"""``G_c(theta)`` with a weak plane -- the ingredient that makes cracks
    deflect.

    Why this is needed, measured rather than assumed
    ------------------------------------------------
    Anisotropic *elasticity* alone does **not** turn cracks. Measured across
    three configurations, microstructure orientation changes the driving force
    by up to **3.29x** and the direction by about **2 degrees**, including in a
    mixed-mode geometry with a genuine -31 degree kink and no restoring
    symmetry. The direction is set by the stress field, which geometry and
    loading dominate.

    Real composites, wood and bone deflect cracks because they have *weak
    planes*: the toughness is low along the fibre or interface. A crack follows
    where ``G / G_c(theta)`` is largest, not where ``G`` is largest. With
    ``G_c`` isotropic there is nothing to pull the direction away from the
    elastic solution -- which is exactly what the measurements show.

    .. math::
        G_c(\theta) = G_{min} + (G_{max} - G_{min})
                       \sin^2(\theta - \theta_{weak})

    so ``G_c`` is smallest along ``theta_weak`` and largest across it. The ratio
    ``G_max/G_min`` is the toughness anisotropy, and it is a **separate**
    material property from the elastic anisotropy ``k_fibre`` -- they need not
    share a direction, though in a fibre composite they usually do.
    """

    def __init__(self, weak_direction: float, g_min: float, g_max: float):
        if g_min <= 0.0:
            raise ValueError(f"g_min must be positive, got {g_min}")
        if g_max < g_min:
            raise ValueError(
                f"g_max ({g_max}) is below g_min ({g_min}); the weak direction "
                f"is the one with the *lower* toughness, so this has the "
                f"anisotropy inverted")
        self.weak_direction = float(weak_direction)
        self.g_min, self.g_max = float(g_min), float(g_max)

    @property
    def ratio(self) -> float:
        return self.g_max / self.g_min

    def __call__(self, theta) -> np.ndarray:
        return self.g_min + (self.g_max - self.g_min) * np.sin(
            np.asarray(theta, dtype=float) - self.weak_direction) ** 2

    def __repr__(self) -> str:
        return (f"AnisotropicToughness(weak={np.degrees(self.weak_direction):.1f}"
                f" deg, g_min={self.g_min:.4g}, ratio={self.ratio:.3g})")


def propagation_direction(J, toughness=None, heading=None, half_cone=1.4,
                          n=721):
    r"""Direction maximising ``G(theta) / G_c(theta)``. Returns radians.

    ``G(theta)`` is taken as the projection ``|G| cos(theta - phi)`` of the
    material force, ``phi`` being its own direction. **That is the standard
    configurational approximation and it is an approximation**: the exact energy
    release for a kinked extension needs the kinked-crack solution (Amestoy &
    Leblond), not a projection. It is exact at ``theta = phi`` and degrades as
    the kink grows, so a large predicted turn should be treated as qualitative.

    With ``toughness=None`` this returns ``phi`` exactly -- the isotropic answer
    -- so the anisotropic path reduces to the existing behaviour rather than
    replacing it.

    ``half_cone`` restricts the search to within that many radians of
    ``heading`` (the current crack direction, defaulting to ``phi``), which
    keeps the crack from turning backwards into its own wake. The default 1.4
    rad is just under 90 degrees.
    """
    J = np.asarray(J, dtype=float)[:2]
    mag = float(np.linalg.norm(J))
    if mag == 0.0:
        raise ValueError("a zero material force has no direction to propagate")
    phi = float(np.arctan2(J[1], J[0]))
    if toughness is None:
        return phi
    ref = phi if heading is None else float(heading)
    th = np.linspace(ref - half_cone, ref + half_cone, n)
    drive = mag * np.cos(th - phi) / toughness(th)
    return float(th[int(np.argmax(drive))])
