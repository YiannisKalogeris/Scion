"""
Warp kernels for 2-D explicit APIC-MPM.

Line-for-line counterparts of ``transfer.py`` and ``stepper.py``. The NumPy
versions stay as the readable reference implementation; these are the fast
path, and ``tests/mpm/test_warp_backend.py`` asserts the two agree.

Why the port is mechanical: the NumPy code was written in "kernel shape" from
the start -- struct-of-arrays state, an explicit ``(n, 9)`` node stencil,
closed-form 2x2 algebra instead of ``np.linalg``. Each ``np.add.at`` scatter
becomes ``wp.atomic_add``; each ``np.einsum`` gather becomes an inner loop over
the 3x3 stencil.

Two differences that are not cosmetic:

* The stencil is *not* materialised. NumPy builds ``(n, 9)`` arrays of weights,
  gradients and node offsets because Python-level loops are unaffordable; a
  Warp kernel recomputes them inline per particle, which is both faster and
  far lighter on memory. At 10^7 particles the NumPy stencil alone would be
  several GB.
* Grid arrays are flat and indexed ``node = (i + pad) * ny + (j + pad)``.
  Reductions go through ``wp.atomic_add``, so P2G is deterministic in result
  but not in floating-point summation order -- two runs can differ in the last
  bits. That is normal for GPU scatter, and is why the tests use a tolerance
  rather than exact equality.

2-D only. The 3-D version is a mechanical substitution of ``vec3``/``mat33``
and a 3x3x3 stencil; kept separate rather than made generic because Warp
kernels are typed and the branching would obscure both.
"""

from __future__ import annotations

import warp as wp

__all__ = [
    "pk1_neohookean",
    "p2g",
    "grid_update",
    "g2p",
    "zero_grid",
]


# ---------------------------------------------------------------------------
# Constitutive law
# ---------------------------------------------------------------------------

@wp.func
def pk1_neohookean(F: wp.mat22, mu: float, lam: float) -> wp.mat22:
    """P = mu (F - F^-T) + lam ln(J) F^-T.

    Mirrors ``materials/hyperelastic_batched.NeoHookeanCompressible``. Adding a
    second model means a second ``wp.func`` plus a branch or a separate kernel;
    Warp cannot dispatch on a Python object.
    """
    J = wp.determinant(F)
    FinvT = wp.transpose(wp.inverse(F))
    return mu * (F - FinvT) + lam * wp.log(J) * FinvT


# ---------------------------------------------------------------------------
# Quadratic B-spline weights, computed inline
# ---------------------------------------------------------------------------

@wp.func
def bspline_w(fx: float) -> wp.vec3:
    """Quadratic B-spline weights for the 3-node stencil along one axis."""
    return wp.vec3(
        0.5 * (1.5 - fx) * (1.5 - fx),
        0.75 - (fx - 1.0) * (fx - 1.0),
        0.5 * (fx - 0.5) * (fx - 0.5),
    )


@wp.func
def bspline_dw(fx: float) -> wp.vec3:
    """Derivatives w.r.t. fx (multiply by inv_dx for physical gradients)."""
    return wp.vec3(fx - 1.5, -2.0 * (fx - 1.0), fx - 0.5)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

@wp.kernel
def zero_grid(grid_m: wp.array(dtype=float),
              grid_mv: wp.array(dtype=wp.vec2)):
    """Reset the grid. The grid carries no state between steps."""
    i = wp.tid()
    grid_m[i] = 0.0
    grid_mv[i] = wp.vec2(0.0, 0.0)


@wp.kernel
def p2g(x: wp.array(dtype=wp.vec2),
        v: wp.array(dtype=wp.vec2),
        F: wp.array(dtype=wp.mat22),
        C: wp.array(dtype=wp.mat22),
        mass: wp.array(dtype=float),
        vol0: wp.array(dtype=float),
        grid_m: wp.array(dtype=float),
        grid_mv: wp.array(dtype=wp.vec2),
        mu: float, lam: float,
        dx: float, inv_dx: float, dt: float,
        ny: int, pad: int):
    """Scatter mass, affine momentum and internal force to the grid.

    Combines ``APIC.p2g`` and ``ExplicitMPMStepper.internal_force`` in one pass,
    so the stencil is built once instead of twice.
    """
    p = wp.tid()

    xp = x[p]
    gx = xp[0] * inv_dx
    gy = xp[1] * inv_dx
    bi = int(wp.floor(gx - 0.5))
    bj = int(wp.floor(gy - 0.5))
    fx = gx - float(bi)
    fy = gy - float(bj)

    wx = bspline_w(fx)
    wy = bspline_w(fy)
    dwx = bspline_dw(fx) * inv_dx
    dwy = bspline_dw(fy) * inv_dx

    # V0 * P * F^T -- identical to V * sigma but without dividing by J
    Fp = F[p]
    P = pk1_neohookean(Fp, mu, lam)
    VsigmaT = vol0[p] * (P * wp.transpose(Fp))

    mp = mass[p]
    vp = v[p]
    Cp = C[p]

    for i in range(3):
        for j in range(3):
            node = (bi + i + pad) * ny + (bj + j + pad)
            weight = wx[i] * wy[j]
            gradw = wp.vec2(dwx[i] * wy[j], wx[i] * dwy[j])

            node_x = float(bi + i) * dx
            node_y = float(bj + j) * dx
            dpos = wp.vec2(node_x - xp[0], node_y - xp[1])

            # affine momentum + internal force impulse
            mom = weight * mp * (vp + Cp * dpos) - dt * (VsigmaT * gradw)

            wp.atomic_add(grid_m, node, weight * mp)
            wp.atomic_add(grid_mv, node, mom)


@wp.kernel
def grid_update(grid_m: wp.array(dtype=float),
                grid_mv: wp.array(dtype=wp.vec2),
                grid_v: wp.array(dtype=wp.vec2),
                gravity: wp.vec2, dt: float,
                dx: float, nx: int, ny: int, pad: int,
                lo_x: float, lo_y: float, hi_x: float, hi_y: float,
                wall_mode: int):
    """Momentum update, body force and separating box walls.

    ``wall_mode``: 0 none, 1 separating, 2 sticky. Node coordinates are
    reconstructed from the flat index rather than stored, which keeps the grid
    to two arrays.
    """
    n = wp.tid()
    m = grid_m[n]
    if m <= 1.0e-12:
        grid_v[n] = wp.vec2(0.0, 0.0)
        return

    vel = grid_mv[n] / m + dt * gravity

    if wall_mode > 0:
        i = n / ny
        j = n - i * ny
        cx = float(i - pad) * dx
        cy = float(j - pad) * dx
        if wall_mode == 2:
            if cx < lo_x or cx > hi_x or cy < lo_y or cy > hi_y:
                vel = wp.vec2(0.0, 0.0)
        else:
            vx = vel[0]
            vy = vel[1]
            if cx < lo_x and vx < 0.0:
                vx = 0.0
            if cx > hi_x and vx > 0.0:
                vx = 0.0
            if cy < lo_y and vy < 0.0:
                vy = 0.0
            if cy > hi_y and vy > 0.0:
                vy = 0.0
            vel = wp.vec2(vx, vy)

    grid_v[n] = vel


@wp.kernel
def g2p(x: wp.array(dtype=wp.vec2),
        v: wp.array(dtype=wp.vec2),
        F: wp.array(dtype=wp.mat22),
        C: wp.array(dtype=wp.mat22),
        grid_v: wp.array(dtype=wp.vec2),
        dx: float, inv_dx: float, dt: float,
        inv_D: float, ny: int, pad: int):
    """Gather velocity and the APIC affine matrix; advance F and x."""
    p = wp.tid()

    xp = x[p]
    gx = xp[0] * inv_dx
    gy = xp[1] * inv_dx
    bi = int(wp.floor(gx - 0.5))
    bj = int(wp.floor(gy - 0.5))
    fx = gx - float(bi)
    fy = gy - float(bj)

    wx = bspline_w(fx)
    wy = bspline_w(fy)
    dwx = bspline_dw(fx) * inv_dx
    dwy = bspline_dw(fy) * inv_dx

    new_v = wp.vec2(0.0, 0.0)
    new_C = wp.mat22(0.0, 0.0, 0.0, 0.0)
    grad_v = wp.mat22(0.0, 0.0, 0.0, 0.0)

    for i in range(3):
        for j in range(3):
            node = (bi + i + pad) * ny + (bj + j + pad)
            weight = wx[i] * wy[j]
            gradw = wp.vec2(dwx[i] * wy[j], wx[i] * dwy[j])

            node_x = float(bi + i) * dx
            node_y = float(bj + j) * dx
            dpos = wp.vec2(node_x - xp[0], node_y - xp[1])

            vi = grid_v[node]
            new_v = new_v + weight * vi
            new_C = new_C + weight * wp.outer(vi, dpos)
            grad_v = grad_v + wp.outer(vi, gradw)

    ident = wp.mat22(1.0, 0.0, 0.0, 1.0)
    F[p] = (ident + dt * grad_v) * F[p]
    v[p] = new_v
    C[p] = inv_D * new_C
    x[p] = xp + dt * new_v
