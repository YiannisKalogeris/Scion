#!/usr/bin/env python3
r"""
Poisson by finite differences, and how to tell whether a scheme works.

Problem
-------
``lap(u) = f`` on a rectangle with Dirichlet data, solved on a uniform grid.
The right-hand side and the boundary values are manufactured from a chosen exact
solution

    u(x, y) = sin(2 pi x / Lx) sin(3 pi y / Ly)

so the error is known at every point and at every resolution. That is the whole
technique: pick the answer, work out what source produces it, and measure how
fast the computed answer approaches it.

Why the order and not the error
-------------------------------
An error is a number; it can be small for the wrong reasons -- a lucky mesh, a
symmetric problem, a solution the scheme happens to represent exactly. The
*rate* at which it falls is a property of the discretisation and nothing else.
Halve the spacing and a second-order scheme divides its error by four, a
fourth-order one by sixteen. A scheme with a mistyped coefficient still
converges, often to a small-looking error, but at the wrong rate or to the wrong
answer, and the slope is what shows it.

That is not hypothetical here. This assembler used to divide the whole 5-point
stencil by ``hx*hy``, which equals ``h^2`` on a square grid and nothing
meaningful on any other. Every square-grid test passed. On a rectangle the
operator was consistent with a different equation and the error sat at 20%
through three refinements without moving -- visible immediately as a slope of
zero, and invisible in any single run.

Three schemes, and one that is not what it looks like
-----------------------------------------------------
``central, order 2``   the classical 5-point stencil
``central, order 4``   a wider stencil; needs boundary data one cell outside
                       the domain, which a manufactured solution supplies free
``compact 9-point``    the Mehrstellen operator, **second** order as written

That last one is the interesting case. The 9-point stencil is advertised as
fourth order and costs nearly twice the 5-point one, but

    A_9 u = lap(u) + (h^2/12) lap^2(u) + O(h^4),

so solving ``A_9 u = f`` leaves an ``O(h^2)`` error like everything else. The
fourth order appears only when the right-hand side is corrected to
``f + (h^2/12) lap f``, which cancels the leading term. The run below shows both
branches: same operator, same mesh, order 2 or order 4 depending purely on how
``f`` is sampled.

Run
---
    python -m examples.fdm.poisson_convergence
    python -m examples.fdm.poisson_convergence --quick
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

from simulacrum.discretization.fdm import FDMAssembler

RESULTS = Path(__file__).resolve().parent / "results"

RESOLUTIONS = [10, 20, 40, 80, 160]
QUICK = [10, 20, 40]

#: The three schemes, plus the compact one in both of its guises.
SCHEMES = [
    ("central, order 2", dict(order=2)),
    ("central, order 4", dict(order=4)),
    ("compact, plain rhs", dict(scheme="compact")),
    ("compact + Mehrstellen", dict(scheme="compact", mehrstellen=True)),
]


def exact(Lx: float, Ly: float):
    """The manufactured solution and the source that produces it."""
    def u(x, y):
        return np.sin(2.0 * np.pi * x / Lx) * np.sin(3.0 * np.pi * y / Ly)

    def f(x, y):
        return -((2.0 * np.pi / Lx) ** 2 + (3.0 * np.pi / Ly) ** 2) * u(x, y)

    return u, f


def solve(n: int, Lx: float, Ly: float, mehrstellen: bool = False, **kwargs):
    """One solve on an ``n x n`` interior grid; returns the relative max error.

    ``grid_shape`` counts interior points, so the spacing that fits ``n`` of
    them into a length ``L`` is ``L / (n + 1)`` -- the boundaries sit one full
    step beyond the first and last unknown and carry data rather than degrees of
    freedom.
    """
    assembler = FDMAssembler(None, (n, n),
                             spacing=(Lx / (n + 1), Ly / (n + 1)), **kwargs)
    u, f = exact(Lx, Ly)
    system = assembler.assemble(source=f, boundary=u, mehrstellen=mehrstellen)
    computed = spsolve(system.matrix.tocsc(), system.rhs)

    points = assembler.coordinates()
    reference = u(*points.T)
    error = float(np.abs(computed - reference).max()
                  / np.abs(reference).max())
    return error, assembler, computed, reference


def observed_order(errors):
    """Slope between successive refinements, each halving the spacing."""
    return [float(np.log2(a / b)) for a, b in zip(errors, errors[1:])]


# ---------------------------------------------------------------------------

def study(Lx: float, Ly: float, resolutions, title: str):
    print(f"\n  {title}  (Lx = {Lx:g}, Ly = {Ly:g})")
    header = "  ".join(f"n={n:<4d}" for n in resolutions)
    print(f"\n    {'scheme':<24} {header}   observed order")

    out = {}
    for label, kwargs in SCHEMES:
        if not np.isclose(Lx, Ly) and kwargs.get("scheme") == "compact":
            continue                    # the compact stencil needs square cells
        errors = [solve(n, Lx, Ly, **kwargs)[0] for n in resolutions]
        rates = observed_order(errors)
        cells = "  ".join(f"{e:8.2e}" for e in errors)
        print(f"    {label:<24} {cells}   "
              + " ".join(f"{r:4.2f}" for r in rates))
        out[label] = (errors, rates)
    return out


def report(square, rectangle):
    print("\n" + "=" * 74)
    print("  What the slopes say")
    print("=" * 74)

    o2 = square["central, order 2"][1][-1]
    o4 = square["central, order 4"][1][-1]
    plain = square["compact, plain rhs"][1][-1]
    fixed = square["compact + Mehrstellen"][1][-1]
    print(f"\n  central order 2        -> {o2:.2f}   (theory 2)")
    print(f"  central order 4        -> {o4:.2f}   (theory 4)")
    print(f"  compact, plain rhs     -> {plain:.2f}   (theory 2, "
          f"despite the nine points)")
    print(f"  compact + Mehrstellen  -> {fixed:.2f}   (theory 4)")

    print("\n  The last two are the same operator on the same grid. The only")
    print("  difference is whether f is sampled pointwise or through the")
    print("  correction stencil, and it is worth two orders of accuracy.")

    r2 = rectangle["central, order 2"][1][-1]
    print(f"\n  On the rectangle the second-order scheme still gives {r2:.2f}.")
    print("  Under the old assembly it gave 0.00: the errors were 2.01e-01,")
    print("  2.00e-01, 2.00e-01 as the grid was refined, because dividing the")
    print("  whole stencil by hx*hy is only the Laplacian when hx == hy. Every")
    print("  square-grid test passed throughout.")


def plot(square, rectangle, resolutions, Lx, Ly):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not installed; skipping the figure")
        return

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.2))

    # the solved field itself
    _, assembler, computed, _ = solve(max(resolutions), 1.0, 1.0)
    field = assembler.recover_fields(computed)["primary"]
    im = ax[0].imshow(field.T, origin="lower", cmap="RdBu_r",
                      extent=[0, 1, 0, 1], aspect="auto")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[0].set_title(f"Solution, {max(resolutions)} x {max(resolutions)}")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")

    h = 1.0 / (np.array(resolutions) + 1)
    for label, (errors, _) in square.items():
        ax[1].loglog(h, errors, "o-", lw=1.3, ms=4, label=label)
    for slope, style in ((2, "--"), (4, ":")):
        ref = square["central, order 2"][0][0] * (h / h[0]) ** slope
        ax[1].loglog(h, ref, style, c="k", lw=1.0,
                     label=f"$h^{slope}$" if slope == 2 else f"$h^{slope}$")
    ax[1].set_xlabel("grid spacing $h$")
    ax[1].set_ylabel("relative max error")
    ax[1].set_title("Square grid: four schemes")
    ax[1].legend(fontsize=7, frameon=False)
    ax[1].grid(alpha=0.3, which="both")

    for label, (errors, _) in rectangle.items():
        ax[2].loglog(h, errors, "s-", lw=1.3, ms=4, label=label)
    broken = np.full(len(h), 0.2)
    ax[2].loglog(h, broken, "x--", c="crimson", lw=1.2,
                 label="old assembly (no convergence)")
    ax[2].set_xlabel("grid spacing $h$")
    ax[2].set_ylabel("relative max error")
    ax[2].set_title(f"Rectangle {Lx:g} x {Ly:g}: the bug")
    ax[2].legend(fontsize=7, frameon=False)
    ax[2].grid(alpha=0.3, which="both")

    fig.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "poisson_convergence.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"\n  figure -> {path}")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    resolutions = QUICK if quick else RESOLUTIONS

    print("Poisson by finite differences: manufactured solution, "
          "measured order")
    print("  u = sin(2 pi x / Lx) sin(3 pi y / Ly),  "
          "Dirichlet data from the same u")

    square = study(1.0, 1.0, resolutions, "Square grid")
    rectangle = study(1.0, 3.0, resolutions, "Rectangular grid, hx != hy")
    report(square, rectangle)
    plot(square, rectangle, resolutions, 1.0, 3.0)
