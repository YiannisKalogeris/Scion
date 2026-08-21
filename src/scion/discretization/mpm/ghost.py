r"""
Ghost stabilisation for the small-cut problem.

The problem
-----------
MPM decouples the material from the grid, so a cell at the edge of the body can
end up holding a single particle in a corner. That cell's nodes then carry a
stiffness proportional to almost nothing, the global system loses coercivity,
and an implicit solve degrades from "slightly ill-conditioned" to "returns
nonsense" to "fails outright" as the cut gets smaller. It is not specific to
MPM: every non-matching-mesh method has it, and it has a name -- the *small-cut
issue*.

It is what makes uGIMP fail on the curved cantilever in
``examples/mpm/micropolar_curved_cantilever.py``: ten particles invert at load
factor 0.87 and step cut-back does not recover, because halving the load does
nothing about a matrix that is nearly singular by geometry.

The fix
-------
Penalise the *jump in the gradient* across the faces of the badly-cut cells.
That couples a poorly supported node to its well-supported neighbours, which
restores coercivity without constraining the solution where it is already well
posed. Following O'Hare, Coombs, Gourgiotis & Augarde, CMAME 449 (2026) 118516,
Section 3.5, for one face of largest side ``h``:

    J = h \int_{dE} G_g^T N^T N G_g dS ,     G_g = G^+ - G^-

where ``G`` is the strain-displacement matrix and ``N`` maps a Voigt strain to
its normal component (the matrix that turns a stress into a traction). The
penalty is then ``gamma * J`` added to the tangent and ``gamma * J u`` to the
residual, with ``gamma`` a fraction of an elastic modulus.

Why it does not change the answer
---------------------------------
``G_g`` annihilates every affine field, because an affine field has a constant
gradient and therefore no jump to penalise. So the stabilisation is invisible to
exactly the states a consistent discretisation must reproduce: the affine patch
test stays exact to round-off with the penalty switched on. That check tests the
*jump* construction and, note, says nothing about ``N`` -- it passes for any
``N`` at all. ``N`` is checked separately against ``sigma n``.

Two limits worth stating
------------------------
**Linear and GIMP bases only.** The penalty is built on the multilinear Lagrange
functions of the grid cells, which is the grid function space for both the
linear hat and GIMPM -- GIMPM changes the particle-to-node quadrature weights,
not the space. A quadratic or cubic B-spline space is ``C^1``, so its gradient
does not jump across a cell face at all and this operator is identically zero;
stabilising there would need the *second* derivative jump. Passing a B-spline
basis raises rather than silently returning nothing.

**It penalises the symmetric gradient**, since ``N`` acts on a Voigt strain.
That is the reference's formulation and it is weaker than penalising the full
gradient jump: a field whose symmetric gradient is continuous while its skew
part jumps is not seen. Kept as specified rather than strengthened, because the
published penalty parameters are calibrated to this form.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import sparse

from .background_grid import CartesianGrid

__all__ = ["ghost_faces", "ghost_penalty", "populated_cells",
           "stabilised_nodes"]

#: 2-point Gauss-Legendre on [0, 1], weights normalised to sum to 1
_GAUSS_X = np.array([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)])
_GAUSS_W = np.array([0.5, 0.5])


def _cell_index(grid: CartesianGrid, x: np.ndarray) -> np.ndarray:
    """Multi-index of the cell holding each point, in node-lattice coordinates.

    A cell is named by its lower-corner node, so cell ``c`` has corner nodes
    ``c + {0,1}^dim`` and the valid range is ``[0, node_shape - 2]``.
    """
    idx = np.floor((x - grid.origin) / grid.dx).astype(np.int64) + grid.pad
    return np.clip(idx, 0, np.array(grid.node_shape, dtype=np.int64) - 2)


def populated_cells(grid: CartesianGrid, x: np.ndarray,
                    lp: np.ndarray | None = None) -> np.ndarray:
    """Boolean array over cells, ``shape = node_shape - 1``.

    ``lp`` is the particle domain half-width. Pass it for a GIMP basis, where a
    particle's support genuinely covers every cell its *domain* overlaps, not
    just the one holding its centre. Ignoring it marks those cells empty, which
    makes their neighbours boundary cells (harmless, more penalty) but leaves
    the cells themselves with no penalised face at all -- which is the case that
    actually needed stabilising.
    """
    shape = tuple(s - 1 for s in grid.node_shape)
    cells = np.zeros(shape, dtype=bool)
    if lp is None:
        cells[tuple(_cell_index(grid, x).T)] = True
        return cells

    lp = np.broadcast_to(np.asarray(lp, dtype=float), x.shape)
    lo = _cell_index(grid, x - lp)
    hi = _cell_index(grid, x + lp)
    span = int((hi - lo).max()) + 1
    for offset in itertools.product(range(span), repeat=grid.dim):
        off = np.array(offset, dtype=np.int64)
        idx = lo + off
        keep = (idx <= hi).all(axis=1)
        if keep.any():
            cells[tuple(idx[keep].T)] = True
    return cells


def ghost_faces(grid: CartesianGrid, x: np.ndarray, lp=None):
    """Faces needing a penalty: ``(axis, cell_lo)`` with ``cell_lo`` the cell on
    the low side of the face along ``axis``.

    A **boundary cell** is populated and face-adjacent to an empty one. The
    penalised faces are the interfaces between a boundary cell and any other
    populated cell -- including other boundary cells, which is what couples a
    whole rim of badly cut cells together rather than leaving each one propped
    against nothing.
    """
    filled = populated_cells(grid, x, lp)
    dim = grid.dim

    # a populated cell touching an empty one across a face
    boundary = np.zeros_like(filled)
    for axis in range(dim):
        for shift in (+1, -1):
            neighbour_empty = ~np.roll(filled, -shift, axis=axis)
            # a cell at the array edge has no neighbour that way; treat the
            # outside as empty, which is what it is
            sl = [slice(None)] * dim
            sl[axis] = slice(0, 1) if shift == -1 else slice(-1, None)
            neighbour_empty[tuple(sl)] = True
            boundary |= filled & neighbour_empty

    faces = []
    for axis in range(dim):
        lo = [slice(None)] * dim
        hi = [slice(None)] * dim
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        a, b = filled[tuple(lo)], filled[tuple(hi)]
        ba, bb = boundary[tuple(lo)], boundary[tuple(hi)]
        # both populated, at least one of them cut
        sel = a & b & (ba | bb)
        for cell in zip(*np.nonzero(sel)):
            faces.append((axis, np.array(cell, dtype=np.int64)))
    return faces


def _corner_offsets(dim: int) -> np.ndarray:
    """``(2**dim, dim)`` local node offsets of a cell, in a fixed order."""
    return np.array(list(itertools.product((0, 1), repeat=dim)),
                    dtype=np.int64)


def _shape_gradients(dim: int, xi: np.ndarray, dx: float) -> np.ndarray:
    """``(2**dim, dim)`` multilinear ``dN_a/dx_j`` at local coordinate ``xi``."""
    corners = _corner_offsets(dim)
    grads = np.empty((len(corners), dim))
    for a, c in enumerate(corners):
        for j in range(dim):
            g = 1.0 if c[j] == 1 else -1.0
            for k in range(dim):
                if k != j:
                    g *= xi[k] if c[k] == 1 else (1.0 - xi[k])
            grads[a, j] = g / dx
    return grads


def _voigt_pairs(dim: int):
    """Shear index pairs in the Voigt order used here: 3-D is
    ``[11, 22, 33, 12, 23, 31]``, 2-D is ``[11, 22, 12]``."""
    return [(0, 1), (1, 2), (2, 0)] if dim == 3 else [(0, 1)]


def _strain_matrix(grads: np.ndarray, dim: int) -> np.ndarray:
    """``(voigt, dim * n_nodes)`` strain-displacement matrix, node-major."""
    n = grads.shape[0]
    voigt = 6 if dim == 3 else 3
    B = np.zeros((voigt, dim * n))
    for a in range(n):
        for j in range(dim):
            B[j, a * dim + j] = grads[a, j]
        for s, (i, j) in enumerate(_voigt_pairs(dim)):
            B[dim + s, a * dim + i] = grads[a, j]
            B[dim + s, a * dim + j] = grads[a, i]
    return B


def _normal_matrix(n: np.ndarray, dim: int) -> np.ndarray:
    """``(dim, voigt)`` mapping a Voigt stress to its traction ``sigma n``.

    In 3-D with the Voigt order above this is

        [[n1, 0,  0,  n2, 0,  n3],
         [0,  n2, 0,  n1, n3, 0 ],
         [0,  0,  n3, 0,  n2, n1]]

    The reference prints ``n3`` in the last entry (its Eq. 148); it must be
    ``n1``, since that slot pairs with the ``31`` shear component and feeds
    ``(sigma n)_3 = ... + sigma_31 n_1``. Derived here rather than transcribed.

    What confirms it is ``N voigt(sigma) == sigma n`` on random symmetric
    tensors and random normals, **not** the affine-kernel test: for an affine
    field the jump ``G_g u`` is already zero, so *any* ``N`` annihilates it and
    that test says nothing about this matrix. Worth stating because the two
    checks look interchangeable and only one of them has any power here.
    """
    voigt = 6 if dim == 3 else 3
    N = np.zeros((dim, voigt))
    for j in range(dim):
        N[j, j] = n[j]
    for s, (i, j) in enumerate(_voigt_pairs(dim)):
        N[i, dim + s] = n[j]
        N[j, dim + s] = n[i]
    return N


def ghost_penalty(grid: CartesianGrid, x: np.ndarray,
                  n_components: int | None = None,
                  lp: np.ndarray | None = None) -> sparse.csr_matrix:
    """Assemble ``J``, the unscaled ghost penalty over a vector field.

    Returns ``(n_nodes * n_components, n_nodes * n_components)``, symmetric
    positive semi-definite, with every affine field in its kernel. Multiply by
    a penalty ``gamma`` (a fraction of an elastic modulus -- the reference uses
    ``lam/100`` for displacement and ``gam/100`` for microrotation) before
    adding it to a tangent, and add ``gamma J u`` to the residual so the two
    stay consistent.

    ``n_components`` defaults to ``grid.dim``; the operator is defined on a
    ``dim``-component field and a micropolar solve applies the *same* ``J``
    twice, once per field, with different penalties.
    """
    from .basis import CubicBSpline, QuadraticBSpline
    if isinstance(grid.basis, (QuadraticBSpline, CubicBSpline)):
        raise NotImplementedError(
            f"ghost stabilisation is formulated on the multilinear grid space, "
            f"which is right for the linear hat and for GIMPM but not for "
            f"{type(grid.basis).__name__}: a B-spline space is C^1, its "
            f"gradient does not jump across a cell face, and this penalty "
            f"would be identically zero. Stabilising there needs the second "
            f"derivative jump.")

    dim = grid.dim
    nc = dim if n_components is None else int(n_components)
    strides = grid._strides
    corners = _corner_offsets(dim)
    n_dof = grid.n_nodes * nc

    faces = ghost_faces(grid, x, lp)
    if not faces:
        return sparse.csr_matrix((n_dof, n_dof))

    # tangential quadrature points on the unit face
    tangential = [np.array(p) for p in
                  itertools.product(_GAUSS_X, repeat=dim - 1)]
    weights = [np.prod(w) for w in
               itertools.product(_GAUSS_W, repeat=dim - 1)]
    area = grid.dx ** (dim - 1)
    h = grid.dx

    rows, cols, vals = [], [], []
    for axis, cell_lo in faces:
        cell_hi = cell_lo.copy()
        cell_hi[axis] += 1
        nodes_lo = (cell_lo + corners) @ strides
        nodes_hi = (cell_hi + corners) @ strides
        union, inv = np.unique(np.concatenate([nodes_lo, nodes_hi]),
                               return_inverse=True)
        map_lo, map_hi = inv[:len(corners)], inv[len(corners):]

        normal = np.zeros(dim)
        normal[axis] = 1.0
        N = _normal_matrix(normal, dim)

        n_u = len(union)
        J = np.zeros((dim * n_u, dim * n_u))
        for t, w in zip(tangential, weights):
            xi = np.empty(dim)
            xi[[k for k in range(dim) if k != axis]] = t
            # the shared face is xi[axis] = 1 seen from the low cell and
            # xi[axis] = 0 seen from the high one
            xi[axis] = 1.0
            B_lo = _strain_matrix(_shape_gradients(dim, xi, grid.dx), dim)
            xi[axis] = 0.0
            B_hi = _strain_matrix(_shape_gradients(dim, xi, grid.dx), dim)

            Bg = np.zeros((B_lo.shape[0], dim * n_u))
            for a in range(len(corners)):
                Bg[:, map_hi[a] * dim:(map_hi[a] + 1) * dim] += \
                    B_hi[:, a * dim:(a + 1) * dim]
                Bg[:, map_lo[a] * dim:(map_lo[a] + 1) * dim] -= \
                    B_lo[:, a * dim:(a + 1) * dim]

            NB = N @ Bg
            J += (w * h * area) * (NB.T @ NB)

        # scatter into every component block of the global field
        dofs = np.concatenate(
            [union * nc + c for c in range(dim)]).reshape(dim, n_u).T.ravel()
        rows.append(np.repeat(dofs, dofs.size))
        cols.append(np.tile(dofs, dofs.size))
        vals.append(J.ravel())

    return sparse.coo_matrix(
        (np.concatenate(vals),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_dof, n_dof)).tocsr()


def stabilised_nodes(grid: CartesianGrid, x: np.ndarray,
                     lp: np.ndarray | None = None) -> np.ndarray:
    """Boolean over nodes: every corner of every populated cell.

    This is the free set a *stabilised* solve should use, and it replaces the
    particle-support threshold rather than supplementing it. The two are in
    direct conflict: a support threshold **removes** a badly cut cell's nodes,
    while ghost stabilisation exists to give exactly those nodes enough
    stiffness to keep. Doing both is worse than doing neither -- the penalty
    then couples a retained node to a dropped one, which reads as a one-sided
    spring pulling the retained node towards zero.

    It replaces the *threshold* and not the requirement of support at all: a
    node with literally zero weight still has no equation, and the penalty does
    not supply one if its cell is isolated, since a face is penalised only when
    **both** its cells are populated. GIMP produces exactly that at a thin tip,
    where a particle domain overlaps a cell no particle centre is in. Callers
    should intersect this with ``support > 0``.
    """
    filled = populated_cells(grid, x, lp)
    nodes = np.zeros(grid.node_shape, dtype=bool)
    for corner in _corner_offsets(grid.dim):
        sl = tuple(slice(c, c + n) for c, n in zip(corner, filled.shape))
        nodes[sl] |= filled
    return nodes.ravel()
