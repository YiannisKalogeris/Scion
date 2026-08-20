"""
Particle <-> grid transfer operators.

PIC, FLIP and APIC differ *only* in these two kernels, which is why they belong
behind one interface rather than being baked into the stepper:

* PIC  -- dissipative; loses the angular momentum stored between particle and node
* FLIP -- transfers the velocity *increment*; noisy, needs blending with PIC
* APIC -- carries an affine velocity field per particle; conserves angular
          momentum and has neither PIC's damping nor FLIP's noise
* PolyPIC -- APIC with a general polynomial velocity per particle instead of an
          affine one. At full degree the mode set spans the whole stencil, so
          the round trip is lossless
* MLSMPM -- APIC transfers, but the *force* uses an MLS gradient kernel in place
          of the basis gradient

APIC is the sensible default and is barely more code than PIC.

References: Jiang et al. (2015), "The affine particle-in-cell method";
Fu, Guo, Gast, Jiang & Teran (2017), "A polynomial particle-in-cell method";
Hu et al. (2018), "A moving least squares material point method".
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .background_grid import CartesianGrid, StencilData
from .particles import ParticleSet


class Transfer(ABC):
    """Strategy for moving momentum between particles and the grid."""

    @abstractmethod
    def p2g(self, particles: ParticleSet, grid: CartesianGrid,
            st: StencilData) -> tuple[np.ndarray, np.ndarray]:
        """Scatter mass and momentum. Returns ``(grid_mass, grid_momentum)``."""

    @abstractmethod
    def g2p(self, particles: ParticleSet, grid: CartesianGrid,
            st: StencilData, grid_v: np.ndarray, dt: float) -> None:
        """Gather velocity back to particles. Updates particle state in place."""

    def force_kernel(self, grid: CartesianGrid, st: StencilData) -> np.ndarray:
        """``(n, k, dim)`` weights the internal force is contracted against.

        The basis gradient for every scheme here except MLS-MPM, which is
        defined by replacing it. Kept on the transfer rather than the stepper
        because it is a property of the particle-grid coupling, and because the
        two steppers would otherwise each need the same branch.
        """
        return st.grad_w


class APIC(Transfer):
    """Affine particle-in-cell."""

    def p2g(self, particles, grid, st):
        grid_m = grid.zeros_scalar()
        grid_mv = grid.zeros_vector()

        wm = st.w * particles.mass[:, None]                     # (n, K)
        np.add.at(grid_m, st.node_id, wm)

        # m_p (v_p + C_p (x_i - x_p))
        affine = particles.v[:, None, :] + np.einsum(
            "pab,pkb->pka", particles.C, st.dpos
        )
        np.add.at(grid_mv, st.node_id, wm[:, :, None] * affine)
        return grid_m, grid_mv

    def g2p(self, particles, grid, st, grid_v, dt):
        vi = grid_v[st.node_id]                                  # (n, K, dim)

        v_new = np.einsum("pk,pka->pa", st.w, vi)
        C_new = grid.basis.apic_inv_D(grid.dx) * np.einsum(
            "pk,pka,pkb->pab", st.w, vi, st.dpos
        )
        grad_v = np.einsum("pka,pkb->pab", vi, st.grad_w)

        eye = np.eye(particles.dim, dtype=particles.dtype)
        particles.F = np.einsum("pij,pjk->pik", eye + dt * grad_v, particles.F)
        particles.v = v_new
        particles.C = C_new
        particles.x = particles.x + dt * v_new


class PIC(Transfer):
    """Particle-in-cell. Kept as a reference: it is APIC with C = 0."""

    def p2g(self, particles, grid, st):
        grid_m = grid.zeros_scalar()
        grid_mv = grid.zeros_vector()
        wm = st.w * particles.mass[:, None]
        np.add.at(grid_m, st.node_id, wm)
        np.add.at(grid_mv, st.node_id, wm[:, :, None] * particles.v[:, None, :])
        return grid_m, grid_mv

    def g2p(self, particles, grid, st, grid_v, dt):
        vi = grid_v[st.node_id]
        v_new = np.einsum("pk,pka->pa", st.w, vi)
        grad_v = np.einsum("pka,pkb->pab", vi, st.grad_w)
        eye = np.eye(particles.dim, dtype=particles.dtype)
        particles.F = np.einsum("pij,pjk->pik", eye + dt * grad_v, particles.F)
        particles.v = v_new
        particles.x = particles.x + dt * v_new


class FLIP(Transfer):
    """FLIP/PIC blend. ``alpha = 1`` is pure FLIP, ``0`` is pure PIC."""

    def __init__(self, alpha: float = 0.95):
        self.alpha = float(alpha)
        self._v_old_grid = None

    def p2g(self, particles, grid, st):
        grid_m = grid.zeros_scalar()
        grid_mv = grid.zeros_vector()
        wm = st.w * particles.mass[:, None]
        np.add.at(grid_m, st.node_id, wm)
        np.add.at(grid_mv, st.node_id, wm[:, :, None] * particles.v[:, None, :])
        active = grid_m > 1.0e-12
        v_old = np.zeros_like(grid_mv)
        v_old[active] = grid_mv[active] / grid_m[active, None]
        self._v_old_grid = v_old
        return grid_m, grid_mv

    def g2p(self, particles, grid, st, grid_v, dt):
        vi = grid_v[st.node_id]
        dvi = (grid_v - self._v_old_grid)[st.node_id]
        v_pic = np.einsum("pk,pka->pa", st.w, vi)
        v_flip = particles.v + np.einsum("pk,pka->pa", st.w, dvi)
        v_new = self.alpha * v_flip + (1.0 - self.alpha) * v_pic

        grad_v = np.einsum("pka,pkb->pab", vi, st.grad_w)
        eye = np.eye(particles.dim, dtype=particles.dtype)
        particles.F = np.einsum("pij,pjk->pik", eye + dt * grad_v, particles.F)
        particles.v = v_new
        particles.x = particles.x + dt * v_pic


class PolyPIC(Transfer):
    r"""Polynomial particle-in-cell (Fu, Guo, Gast, Jiang & Teran 2017).

    APIC carries one affine matrix per particle; PolyPIC carries the
    coefficients of a general polynomial in the node offset ``z = x_i - x_p``,

    .. math::  v(z) = \sum_r c_r\, s_r(z), \qquad s_r(z) = \prod_eta z_eta^{i_eta}

    recovered each step by the mass-weighted least-squares projection of the
    stencil's nodal velocities onto that span. ``degree=1`` gives the
    multilinear modes ``{1, x, y, xy}``; ``degree=2`` gives all nine in 2-D,
    which is the full dimension of a 3-node stencil, so the round trip is
    lossless and the transfer stops dissipating altogether.

    Why a Gram solve rather than the usual diagonal shortcut
    --------------------------------------------------------
    The literature projects mode by mode, ``c_r = sum_i w S_r v_i / sum_i w
    S_r^2``, which is exact only if the modes are mass-orthogonal. Measured on
    Scion's own bases, they are not: orthogonality of ``x`` against ``x^2 - D``
    needs the third weight moment ``sum_i w z^3`` to vanish, and for the
    **quadratic** B-spline it does not -- it is nonzero *and* varies with
    sub-cell position. The cubic B-spline's third moment does vanish.

    So the shortcut is basis-dependent and silently wrong on the default basis.
    This solves the small ``R x R`` system per particle instead: exact for any
    basis and any mode set, at the cost of a batched solve of a 4x4 or 9x9.

    Modes are formed in units of ``dx``. In physical units they span 1 to
    ``dx^4``, which for a typical grid is fourteen orders of magnitude and a
    Gram matrix no solver will touch.
    """

    #: modes whose scaled eigenvalue falls below this are dropped from the
    #: projection rather than inverted
    RCOND = 1.0e-8

    #: degree 2 and above are implemented but do not run stably -- see the
    #: class docstring. Set True to experiment anyway.
    ALLOW_HIGH_DEGREE = False

    def __init__(self, degree: int = 1, rcond: float | None = None,
                 allow_high_degree: bool | None = None):
        if degree < 1:
            raise ValueError("degree must be at least 1 (multilinear)")
        allow = (self.ALLOW_HIGH_DEGREE if allow_high_degree is None
                 else bool(allow_high_degree))
        if degree > 1 and not allow:
            raise NotImplementedError(
                f"PolyPIC degree={degree} is not supported. The multilinear "
                "modes (degree=1) are verified; the quadratic and higher modes "
                "destabilise every run tried, at every CFL number down to "
                "1/8 of the APIC step, with the Eq. 12 configuration pull-back "
                "in place and the mode Gram matrix pseudo-inverted. The cause "
                "has not been isolated, so it is refused rather than shipped "
                "as a scheme that quietly blows up. Pass "
                "allow_high_degree=True to experiment."
            )
        self.degree = int(degree)
        self.rcond = self.RCOND if rcond is None else float(rcond)
        self._exponents = None
        #: mean number of modes that survived truncation on the last gather
        self.effective_rank = None

    def _raw_modes(self, z):
        """``(n, k, R)`` monomials of the (already non-dimensional) offset."""
        out = np.empty(z.shape[:2] + (len(self._exponents),), dtype=z.dtype)
        for r, e in enumerate(self._exponents):
            m = np.ones(z.shape[:2], dtype=z.dtype)
            for a, power in enumerate(e):
                if power:
                    m = m * z[:, :, a] ** power
            out[:, :, r] = m
        return out

    def _modes(self, grid, st):
        """``(n, k, R)`` mode values at the current offsets, in units of dx."""
        import itertools
        if self._exponents is None:
            self._exponents = list(itertools.product(
                range(self.degree + 1), repeat=grid.dim))
        return self._raw_modes(st.dpos * grid.inv_dx)

    @property
    def n_modes(self) -> int:
        return len(self._exponents) if self._exponents is not None else -1

    def p2g(self, particles, grid, st):
        grid_m = grid.zeros_scalar()
        grid_mv = grid.zeros_vector()
        wm = st.w * particles.mass[:, None]
        np.add.at(grid_m, st.node_id, wm)

        c = getattr(particles, "poly_c", None)
        pull = getattr(particles, "poly_pullback", None)
        if c is None or c.shape[-1] != grid.dim or pull is None:
            # Cold start. Seed the polynomial from the affine state the particle
            # already carries rather than dropping to PIC momentum: falling back
            # to the constant mode discards whatever spin was prescribed, which
            # costs ~2% of the angular momentum on the first step and is exactly
            # the loss PolyPIC exists to avoid.
            v_at_nodes = (particles.v[:, None, :]
                          + np.einsum("pab,pkb->pka", particles.C, st.dpos))
        else:
            # Fu et al. Eq. 12: the stored polynomial is anchored at the
            # particle's *previous* position, and the particle has since moved.
            # Evaluating the modes at the current offset instead would shift
            # their origin, and a shift mixes the modes into one another --
            # (z + d)^2 = z^2 + 2dz + d^2 -- so the coefficients would be
            # reinterpreted in a basis they were never fitted in. Pull the node
            # offset back through the affine motion first.
            z = np.einsum("pab,pkb->pka", pull, st.dpos) * grid.inv_dx
            S = self._raw_modes(z)
            v_at_nodes = np.einsum("pkr,pra->pka", S, c)
        np.add.at(grid_mv, st.node_id, wm[:, :, None] * v_at_nodes)
        return grid_m, grid_mv

    def g2p(self, particles, grid, st, grid_v, dt):
        vi = grid_v[st.node_id]                              # (n, K, dim)
        S = self._modes(grid, st)                            # (n, K, R)

        # Jacobi-scale to unit mass-norm first, so the truncation threshold
        # below means the same thing for every mode. Even in dx units the raw
        # monomials differ in norm by orders of magnitude.
        norm = np.sqrt(np.maximum(
            np.einsum("pk,pkr,pkr->pr", st.w, S, S), 1.0e-300))
        Sn = S / norm[:, None, :]

        gram = np.einsum("pk,pkr,pks->prs", st.w, Sn, Sn)    # SPD, unit diag
        rhs = np.einsum("pk,pkr,pka->pra", st.w, Sn, vi)

        # Truncated pseudo-inverse rather than a solve. At full degree the
        # monomial modes are close to linearly dependent -- measured condition
        # 1e8 on a 3-node stencil -- and a straight solve returns coefficients
        # whose error compounds through the momentum update until J goes
        # negative and the run NaNs. Dropping the offending directions instead
        # degrades PolyPIC gracefully towards a lower-order transfer, which is
        # a worse scheme but still a scheme.
        evals, evecs = np.linalg.eigh(gram)
        cutoff = self.rcond * evals.max(axis=1, keepdims=True)
        keep = evals > cutoff
        inv = np.where(keep, 1.0 / np.where(keep, evals, 1.0), 0.0)
        self.effective_rank = float(keep.sum(axis=1).mean())

        c = np.einsum("prs,ps,pqs,pqa->pra", evecs, inv, evecs, rhs)
        c = c / norm[:, :, None]

        particles.poly_c = c
        # every non-constant mode vanishes at z = 0, so the particle velocity
        # is the constant coefficient
        particles.v = c[:, 0, :]

        grad_v = np.einsum("pka,pkb->pab", vi, st.grad_w)
        eye_d = np.eye(particles.dim, dtype=particles.dtype)
        particles.F = np.einsum("pij,pjk->pik", eye_d + dt * grad_v, particles.F)
        particles.x = particles.x + dt * particles.v

        # (I + dt C)^-1 for the next P2G, with C the affine part of the fit --
        # the coefficients of the degree-one modes, restored to physical units
        # The modes are built on z/dx, so the degree-one coefficients are
        # velocities per *cell*, not per unit length: the affine matrix is
        # c/dx. Without the inv_dx this pull-back is wrong by a factor of the
        # grid spacing, which for a typical grid is one to two orders.
        n, d = particles.n, particles.dim
        C = np.zeros((n, d, d), dtype=c.dtype)
        for a in range(d):
            e = [0] * d
            e[a] = 1
            C[:, :, a] = c[:, self._exponents.index(tuple(e)), :] * grid.inv_dx
        particles.C = C
        particles.poly_pullback = np.linalg.inv(eye_d + dt * C)


class MLSMPM(APIC):
    r"""Moving least squares MPM (Hu et al. 2018).

    APIC transfers, with the internal force contracted against an MLS gradient
    kernel instead of the basis gradient:

    .. math::  
abla w_{ip} \;\longrightarrow\; w_{ip}\, D_p^{-1} (x_i - x_p)

    which is what lets P2G fuse the affine momentum and the stress term into one
    matrix. Everything else is APIC, so this subclasses it.

    Measured, on Scion's bases
    --------------------------
    Both kernels reproduce the gradient of a **linear** nodal field exactly, to
    1e-15 -- that is the MLS reproducing property and the reason the
    substitution is legitimate. On a quadratic field they differ by 0.43% for
    the quadratic B-spline, and by **5e-15 for the cubic**, i.e. not at all.
    The cubic's third weight moment vanishes, so its basis gradient already
    agrees with the MLS one to the next order.

    So MLS-MPM is a distinct scheme on the quadratic basis and is
    indistinguishable from standard MPM on the cubic. Worth knowing before
    ranking the two against each other.
    """

    def force_kernel(self, grid, st):
        return st.w[:, :, None] * grid.basis.apic_inv_D(grid.dx) * st.dpos


TRANSFER_REGISTRY: dict[str, type[Transfer]] = {
    "pic": PIC,
    "flip": FLIP,
    "apic": APIC,
    "polypic": PolyPIC,
    "mls": MLSMPM,
    "mls-mpm": MLSMPM,
}


def get_transfer(name: str, **kwargs) -> Transfer:
    if name not in TRANSFER_REGISTRY:
        raise NotImplementedError(f"Unknown transfer scheme '{name}'")
    return TRANSFER_REGISTRY[name](**kwargs)
