r"""
Reducing a tensor history to the scalar one a life curve can use.

An S-N or eps-N curve is measured on a uniaxial specimen and takes one number
per cycle. A finite element analysis produces six. Something has to bridge that,
and every bridge in this module is an approximation whose failure modes are
worth knowing before the answer is quoted.

Sign is the whole problem
-------------------------
The obvious reduction, von Mises, is positive by construction. Feed a fully
reversed uniaxial history through it and the result is a rectified signal: the
range halves and the cycle count doubles, and rainflow -- which cannot know any
better -- reports twice as many cycles at half the amplitude. On a curve with
slope ``m = 3`` that is an eight-fold error in damage per cycle against a
two-fold error in count, so the life comes out roughly four times too long. In
the right direction to be dangerous.

:func:`signed_von_mises` is the fix and the default: the magnitude of von Mises,
carrying the sign of the stress state. It restores the mean stress and the
correct cycle count for any history that stays close to proportional loading.

What none of this handles
-------------------------
Under **non-proportional** loading -- where the principal directions rotate
during the cycle -- no scalar reduction is correct, and the error is not a
factor of two. The crack plane changes during the cycle, the material hardens
extra because of the rotation, and lives come out unconservative by an order of
magnitude. The remedy is a critical-plane method (Findley, Fatemi-Socie,
Brown-Miller), which searches over candidate planes and counts on each; that is
a genuinely different calculation, not a different reducer, and it is absent
here rather than approximated.

Check :func:`non_proportionality` before trusting any of these on a multiaxial
history. It costs one eigen-decomposition per point and tells you whether the
question is the one this module can answer.

Plane strain is a trap
----------------------
A 2-D Voigt stress from the assembler is ``[s11, s22, s12]``. ``sigma_33`` is
not in it and is not zero: in plane strain it carries the out-of-plane
constraint, and it enters ``J2`` directly. Leaving the slot empty is not a small
approximation, and the size of the error depends on the state rather than on a
fixed factor. At ``nu = 0.3``, an in-plane uniaxial stress comes out 12% too
*high*; an equibiaxial one comes out 2.5 times too high, because the true
deviator there is almost entirely the difference against ``sigma_33`` and
dropping it leaves the full in-plane stress instead.

Conservative in both cases, which is exactly why it survives review. It is still
the wrong number, and on a curve with slope ``m = 3`` a factor of 2.5 in stress
is a factor of 15 in life. :func:`voigt_to_tensor` therefore refuses a 2-D input
unless it is told how to fill the slot.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from ..materials.notation import from_mandel
from ..materials.notation import voigt_strain_to_tensor as _voigt_strain_to_tensor

__all__ = [
    "voigt_to_tensor",
    "voigt_strain_to_tensor",
    "mandel_to_tensor",
    "plane_strain_sigma_33",
    "von_mises",
    "signed_von_mises",
    "max_principal",
    "max_shear",
    "equivalent_strain",
    "non_proportionality",
    "EQUIVALENT_REGISTRY",
    "get_equivalent",
]


# ---------------------------------------------------------------------------
# Getting to full 3-D tensors
# ---------------------------------------------------------------------------

def plane_strain_sigma_33(sigma_11, sigma_22, nu: float) -> np.ndarray:
    """``sigma_33 = nu (sigma_11 + sigma_22)``. **Elastic plane strain only.**

    Under plastic flow this is wrong. Plastic straining is isochoric, so the
    out-of-plane stress drifts from the elastic value towards the mean of the
    in-plane stresses -- the effective ``nu`` runs from its elastic value to
    1/2 as the plastic strain grows -- and the error lands squarely in ``J2``,
    which is what the fatigue reduction is about to take.

    For an elastoplastic analysis the exact value is available and costs
    nothing, because the stress is always ``C_el : (eps - eps_p)`` and both
    parts are already stored::

        sigma_mandel = (voigt_strain_to_mandel(eps, dim=2)
                        - state.eps_p) @ model.C_el.T

    which gives all six components including ``sigma_33``. Use that; use this
    function only when the analysis really was elastic.
    """
    s11 = np.asarray(sigma_11, dtype=float)
    s22 = np.asarray(sigma_22, dtype=float)
    return float(nu) * (s11 + s22)


def voigt_to_tensor(sigma, dim: int = 3, sigma_33=None,
                    nu: float | None = None) -> np.ndarray:
    """Voigt stress ``(..., 6)`` or ``(..., 3)`` to tensors ``(..., 3, 3)``.

    In 3-D the input is ``[s11, s22, s33, s23, s13, s12]``.

    In 2-D it is ``[s11, s22, s12]`` and the out-of-plane component is missing.
    Supply either ``sigma_33`` -- an array broadcastable to the leading shape,
    the exact value from an elastoplastic run -- or ``nu``, which applies the
    elastic plane-strain relation. Neither is a default: filling the slot with
    zero would be plane *stress*, which is a different problem, and it would
    change the answer silently.
    """
    s = np.asarray(sigma, dtype=float)
    lead = s.shape[:-1]
    out = np.zeros(lead + (3, 3))

    if dim == 3:
        if s.shape[-1] != 6:
            raise ValueError(f"3-D Voigt stress needs 6 components, got "
                             f"{s.shape[-1]}")
        out[..., 0, 0] = s[..., 0]
        out[..., 1, 1] = s[..., 1]
        out[..., 2, 2] = s[..., 2]
        out[..., 1, 2] = out[..., 2, 1] = s[..., 3]
        out[..., 0, 2] = out[..., 2, 0] = s[..., 4]
        out[..., 0, 1] = out[..., 1, 0] = s[..., 5]
        return out

    if dim != 2:
        raise ValueError(f"dim must be 2 or 3, got {dim}")
    if s.shape[-1] != 3:
        raise ValueError(f"2-D Voigt stress needs 3 components "
                         f"[s11, s22, s12], got {s.shape[-1]}")
    if sigma_33 is None and nu is None:
        raise ValueError(
            "a 2-D Voigt stress does not carry sigma_33, and in plane strain "
            "it is neither zero nor negligible: it enters J2 and therefore the "
            "equivalent stress the fatigue curve will see. Pass sigma_33 (the "
            "exact value, available from an elastoplastic state as "
            "C_el:(eps - eps_p)) or nu (the elastic plane-strain relation).")

    out[..., 0, 0] = s[..., 0]
    out[..., 1, 1] = s[..., 1]
    out[..., 0, 1] = out[..., 1, 0] = s[..., 2]
    out[..., 2, 2] = (plane_strain_sigma_33(s[..., 0], s[..., 1], nu)
                      if sigma_33 is None
                      else np.broadcast_to(np.asarray(sigma_33, dtype=float),
                                           lead))
    return out


#: Voigt *strain* to tensors. Re-exported from
#: :mod:`simulacrum.materials.notation`, which owns every Voigt conversion in
#: the library, rather than reimplemented here -- the engineering-shear halving
#: and the ``eps_33 = 0`` plane-strain convention are exactly the details that
#: should have one tested implementation. Contrast :func:`voigt_to_tensor`
#: above, which is not a pure conversion: it carries the fatigue-specific
#: refusal to guess ``sigma_33``.
voigt_strain_to_tensor = _voigt_strain_to_tensor


def mandel_to_tensor(sigma) -> np.ndarray:
    """Mandel ``(..., 6)`` to tensors ``(..., 3, 3)``.

    Thin wrapper over :func:`~simulacrum.materials.notation.from_mandel` that
    accepts arbitrary leading dimensions, so a ``(n_points, n_steps, 6)``
    history converts in one call. The ``sqrt(2)`` bookkeeping stays in
    ``notation``; there is one implementation of it and this is not a second.
    """
    s = np.asarray(sigma, dtype=float)
    lead = s.shape[:-1]
    return from_mandel(s.reshape(-1, 6)).reshape(lead + (3, 3))


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

def _deviator(t: np.ndarray) -> np.ndarray:
    tr = np.trace(t, axis1=-2, axis2=-1) / 3.0
    return t - tr[..., None, None] * np.eye(3)


def von_mises(sigma) -> np.ndarray:
    """``q = sqrt(3/2 s:s)``, ``(...)``. Always non-negative.

    Read the module docstring before using this one directly on a history: it
    rectifies the signal, and the resulting life is optimistic by roughly a
    factor of four on fully reversed loading. :func:`signed_von_mises` is what
    you want almost always.
    """
    s = _deviator(np.asarray(sigma, dtype=float))
    return np.sqrt(1.5 * np.einsum("...ij,...ij->...", s, s))


def _sign_of(t: np.ndarray, mode: str) -> np.ndarray:
    """Sign to attach to a positive invariant."""
    if mode == "hydrostatic":
        s = np.sign(np.trace(t, axis1=-2, axis2=-1))
        # a deviatoric state (pure shear, fully developed plastic flow) has no
        # hydrostatic sign at all; fall back rather than emit a zero, which
        # would collapse the cycle to nothing
        w = np.linalg.eigvalsh(t)
        alt = np.sign(np.take_along_axis(
            w, np.argmax(np.abs(w), axis=-1)[..., None], axis=-1)[..., 0])
        return np.where(s != 0.0, s, np.where(alt != 0.0, alt, 1.0))
    if mode == "principal":
        w = np.linalg.eigvalsh(t)
        s = np.sign(np.take_along_axis(
            w, np.argmax(np.abs(w), axis=-1)[..., None], axis=-1)[..., 0])
        return np.where(s != 0.0, s, 1.0)
    raise ValueError(f"sign must be 'hydrostatic' or 'principal', got '{mode}'")


def signed_von_mises(sigma, sign: str = "hydrostatic") -> np.ndarray:
    """von Mises carrying the sign of the stress state, ``(...)``.

    ``sign="hydrostatic"`` takes the sign of ``tr(sigma)``, which is the usual
    meaning of "signed von Mises" and the default. A purely deviatoric state has
    no hydrostatic sign, so those points fall back to the principal rule rather
    than being reported as zero -- otherwise fully developed plastic flow, which
    is isochoric, would count as no cycle at all.

    ``sign="principal"`` takes the sign of the principal stress largest in
    magnitude. It is better behaved in shear and worse near equibiaxial states,
    where the two rules can disagree on consecutive steps and manufacture a
    spurious reversal. Neither rule is right in general; they differ only where
    the reduction itself is questionable.
    """
    t = np.asarray(sigma, dtype=float)
    return _sign_of(t, sign) * von_mises(t)


def max_principal(sigma) -> np.ndarray:
    """Largest principal stress, ``(...)``.

    The right reduction for brittle materials and for weld assessment, where the
    crack plane is normal to the maximum tension. It ignores the other two
    principals entirely, so it under-predicts damage in a shear-dominated state.
    """
    return np.linalg.eigvalsh(np.asarray(sigma, dtype=float))[..., -1]


def max_shear(sigma) -> np.ndarray:
    """Tresca shear, ``(sigma_1 - sigma_3)/2``, ``(...)``. Non-negative.

    Unsigned, and so subject to the same rectification problem as
    :func:`von_mises`. Useful as a diagnostic and as the driving quantity for
    shear-based critical-plane criteria, which are not implemented here.
    """
    w = np.linalg.eigvalsh(np.asarray(sigma, dtype=float))
    return 0.5 * (w[..., -1] - w[..., 0])


def equivalent_strain(eps, nu: float = 0.5, sign: str = "hydrostatic"
                      ) -> np.ndarray:
    r"""Signed von Mises equivalent strain, ``(...)``.

        ``eps_eq = sqrt(3/2 e:e) / (1 + nu)``

    normalised so that a uniaxial state of strain ``eps`` returns ``eps``
    exactly, for whatever ``nu`` describes it.

    That normalisation is the whole difficulty. Elastic strain has
    ``nu = nu_elastic``; fully plastic strain is isochoric and has ``nu = 1/2``;
    a real elastoplastic history is somewhere between and moves during the
    load. Passing 1/2 for an elastic field under-reads the equivalent strain by
    ``(1 + nu_el)/1.5``, about 13% at ``nu = 0.3``, and on a strain-life curve
    13% of strain is a large multiple of life.

    Which is why this default is 1/2 and not 0.3: strain-life is used where the
    strains are plastic. If your history is elastic, say so, or use
    :func:`max_principal` on the strain tensor, which needs no such constant.
    """
    e = np.asarray(eps, dtype=float)
    d = _deviator(e)
    mag = np.sqrt(1.5 * np.einsum("...ij,...ij->...", d, d)) / (1.0 + float(nu))
    return _sign_of(e, sign) * mag


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def non_proportionality(sigma) -> np.ndarray:
    """How much the principal directions rotate over a history, in ``[0, 1]``.

    ``sigma`` is ``(..., n_steps, 3, 3)``; the result is ``(...)``, one number
    per point. Zero means the principal axes never moved -- proportional
    loading, and every reducer in this module is defensible. Nonzero means they
    did, the reduction is no longer justified, and the life will be
    over-predicted, typically by an order of magnitude.

    Read the scale from a calibration point, not from the ``[0, 1]`` range. The
    canonical non-proportional case -- axial and torsional loading 90 degrees
    out of phase at equal von Mises amplitude -- scores only **0.11**, because
    the dominant principal direction of an axial-torsion state sweeps 45
    degrees rather than 90. In-phase loading of the same two components scores
    zero to machine precision. So 0.11 is severe, not mild, and anything above
    roughly 0.05 is worth a critical-plane check.

    Measured as one minus the mean absolute alignment between the dominant
    principal direction at each step and the direction at the most heavily
    loaded step, weighted by the von Mises stress so that near-zero steps, whose
    principal directions are noise, do not dominate.

    "Dominant" means largest in *magnitude*, not largest signed. Under a
    reversed uniaxial history the largest signed principal is the loading
    direction in tension and one of the two degenerate zero directions in
    compression, so ranking by sign would report a rotation through 90 degrees
    every half cycle -- for the most proportional history there is.

    A screening number, not a criterion. Use it to decide whether the question
    being asked is the one this module can answer.
    """
    s = np.asarray(sigma, dtype=float)
    q = von_mises(s)                                   # (..., n_steps)
    w, v = np.linalg.eigh(s)                           # columns are directions
    j = np.argmax(np.abs(w), axis=-1)                  # (..., n_steps)
    d = np.take_along_axis(v, j[..., None, None], axis=-1)[..., 0]

    k = np.argmax(q, axis=-1)                          # busiest step
    ref = np.take_along_axis(d, k[..., None, None], axis=-2)   # (..., 1, 3)

    align = np.abs(np.einsum("...si,...si->...s", d, np.broadcast_to(ref, d.shape)))
    wgt = q / np.maximum(q.sum(axis=-1, keepdims=True), 1e-300)
    return 1.0 - np.sum(align * wgt, axis=-1)


# ---------------------------------------------------------------------------

#: Reductions from a tensor history to a scalar one. Entries are plain
#: callables ``(..., 3, 3) -> (...)``; :func:`get_equivalent` binds the
#: parameterised ones.
EQUIVALENT_REGISTRY = {
    "signed-von-mises": signed_von_mises,
    "von-mises": von_mises,
    "max-principal": max_principal,
    "max-shear": max_shear,
    "tresca": max_shear,
    "equivalent-strain": equivalent_strain,
}


def get_equivalent(name: str, **kwargs):
    """Look up a reduction, binding any keyword arguments.

    ``get_equivalent("signed-von-mises", sign="principal")`` returns a callable
    of one argument, which is what the assessment driver expects.
    """
    key = name.lower().replace("_", "-")
    if key not in EQUIVALENT_REGISTRY:
        raise NotImplementedError(
            f"Unknown equivalent measure '{name}'. Available: "
            f"{sorted(set(EQUIVALENT_REGISTRY))}. Critical-plane criteria "
            f"(Findley, Fatemi-Socie, Brown-Miller) are absent by design: they "
            f"count cycles on each candidate plane rather than reducing the "
            f"history first, so they are a different calculation and not a "
            f"member of this registry.")
    fn = EQUIVALENT_REGISTRY[key]
    return partial(fn, **kwargs) if kwargs else fn
