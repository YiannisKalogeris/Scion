r"""
Eshelby stress, pseudomomentum, and material (configurational) forces.

Every other stress measure in ``materials/`` answers a question on the *spatial*
manifold: what force does the material exert where it currently is. These answer
the complementary question on the *material* manifold: what force is conjugate
to moving a piece of material to a different place in the **reference**
configuration.

That distinction is not a formality. The Cauchy and Piola stresses vanish for an
unloaded body; the Eshelby stress does not vanish at a crack tip, an inclusion,
or a stiffness gradient, because those are inhomogeneities of the reference
configuration itself. It is the natural object for anything asking *where the
material wants to rearrange*: crack driving force, defect and interface
mechanics, shape sensitivity, mesh error estimation and *r*-adaptivity.

Definitions
-----------
For a hyperelastic body with strain-energy density ``W(F; X)`` per unit
reference volume, reference density ``rho_0``, and kinetic energy density
``K = rho_0 |v|^2 / 2`` (also per unit *reference* volume),

.. math::
    \Sigma = (W - K) I - F^T P, \qquad \mathcal{P} = -\rho_0 F^T v

with ``P = dW/dF`` the first Piola-Kirchhoff stress. ``Sigma`` is the Eshelby
energy-momentum tensor and ``P`` (script) the pseudomomentum, or canonical
momentum, density. Their balance on the material manifold is

.. math::
    \operatorname{Div}\Sigma + \mathcal{G}^{\rm inh}
        = \partial\mathcal{P}/\partial t\big|_X

where ``G^inh`` collects every *explicit* dependence of the energetics on ``X``.
In a homogeneous body the static case reduces to ``Div Sigma = 0``, which is the
path-independence from which Rice's J-integral follows.

Two traps
---------
**Sigma is not symmetric.** ``F^T P = C S`` with ``C = F^T F`` and ``S`` the
second Piola-Kirchhoff stress; a product of two symmetric matrices is symmetric
only when they commute, which happens for isotropic materials and not in
general. So the contraction index matters and is fixed by the traction form
``Sigma_{KL} N_L`` -- the *second* index contracts with the normal or the basis
gradient. :func:`material_force` does that and nothing else.

**Kinetic energy per reference volume.** ``K`` here is ``rho_0 |v|^2 / 2``, not
``rho |v|^2 / 2``. Passing the spatial density is wrong by a factor of ``J`` and
will look plausible, because it is only a few percent off until the deformation
is large.

Scope
-----
Hyperelasticity. For a dissipative material the split of ``Sigma`` into
energetic and dissipative parts is not unique, and this module does not choose
one: pass the elastic ``W`` and you get the energetic part only. That is a
modelling decision the caller has to make deliberately, not a default.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "kinetic_energy_density",
    "eshelby_stress",
    "eshelby_stress_small_strain",
    "pseudomomentum",
    "total_pseudomomentum",
    "material_force",
    "configurational_torque",
    "micropolar_eshelby_stress",
    "j_integral",
]


def kinetic_energy_density(rho0, v: np.ndarray) -> np.ndarray:
    """``K = rho_0 |v|^2 / 2`` per unit **reference** volume. ``(n, d) -> (n,)``.

    ``rho0`` may be a scalar or ``(n,)``.
    """
    v = np.asarray(v)
    return 0.5 * np.asarray(rho0) * np.einsum("pi,pi->p", v, v)


def eshelby_stress(energy: np.ndarray, F: np.ndarray, P: np.ndarray,
                   kinetic: np.ndarray | None = None) -> np.ndarray:
    r"""``Sigma = (W - K) I - F^T P``. ``(n,), (n,d,d), (n,d,d) -> (n,d,d)``.

    Parameters
    ----------
    energy  : ``(n,)`` strain-energy density per unit reference volume
    F       : ``(n, d, d)`` deformation gradient
    P       : ``(n, d, d)`` first Piola-Kirchhoff stress
    kinetic : ``(n,)`` kinetic energy density, also per unit reference volume.
              Omit for the static Eshelby stress.

    Static by default on purpose: the nodal material force of a quasi-static
    problem uses ``W I - F^T P``, and silently subtracting a kinetic term that
    the caller did not ask for would change the answer without saying so.
    """
    F = np.asarray(F)
    P = np.asarray(P)
    scalar = np.asarray(energy, dtype=float)
    if kinetic is not None:
        scalar = scalar - np.asarray(kinetic, dtype=float)

    d = F.shape[1]
    out = -np.einsum("pki,pkj->pij", F, P)          # -(F^T P)_{ij}
    idx = np.arange(d)
    out[:, idx, idx] += scalar[:, None]
    return out


def eshelby_stress_small_strain(energy: np.ndarray, H: np.ndarray,
                                stress: np.ndarray) -> np.ndarray:
    r"""``Sigma = W I - H^T sigma``, the classical LEFM energy-momentum tensor.

    ``H = Grad u`` is the displacement gradient and ``sigma`` the Cauchy stress.
    This is the object Rice's J-integral is normally written with, and for
    linear-elastic fracture it is the one to use.

    Why, when :func:`eshelby_stress` is the more general object
    -----------------------------------------------------------
    Both give the same ``J``; they differ by a term of order ``sigma`` that
    integrates to zero over a closed contour. But ``sigma`` near a crack tip is
    *five orders of magnitude larger than J itself*, so in the finite-strain
    form that term is a near-total cancellation, and the quadrature error left
    behind is comparable to the answer. Measured on the Williams K-field: this
    form is accurate to 0.3% on a coarse ring and path-independent to five
    figures, while the finite-strain form needs roughly ten times the
    quadrature to reach 0.4% and drifts with contour radius until it gets there.

    Use :func:`eshelby_stress` when the problem is genuinely finite-strain, and
    this when the constitutive model is linear elasticity.
    """
    H = np.asarray(H)
    out = -np.einsum("pki,pkj->pij", H, np.asarray(stress))
    d = H.shape[1]
    idx = np.arange(d)
    out[:, idx, idx] += np.asarray(energy, dtype=float)[:, None]
    return out


def micropolar_eshelby_stress(energy: np.ndarray, F: np.ndarray,
                              P: np.ndarray, wryness: np.ndarray,
                              couple: np.ndarray,
                              kinetic: np.ndarray | None = None) -> np.ndarray:
    r"""``Sigma_KL = (W - K) d_KL - F_iK P_iL - Gamma_gK S_gL``.

    The energy-momentum tensor of a micropolar continuum. ``wryness`` is
    ``Gamma`` and ``couple`` the **Biot** couple-stress ``S = dW/dGamma``, both
    material-frame objects, as produced by
    :class:`~scion.materials.micropolar.MicropolarHooke`.

    Derived, not transcribed, and the derivation is worth keeping because it
    removes an ambiguity. Noether for material translations gives

        Sigma_KL = W d_KL - sum_A (dA/dX_K)(dW/d(dA/dX_L))

    over the fields ``A``. The microrotation enters through ``dQ/dX``, and since
    ``Gamma_gp = -1/2 e_gth Q_pt dQ_ph/dX_p``,

        (dQ_ab/dX_K) dW/d(dQ_ab/dX_L)
            = S_gL [ -1/2 e_gtb Q_at dQ_ab/dX_K ] = S_gL Gamma_gK

    -- the bracket *is* ``Gamma`` by its own definition, verified exactly. So
    the couple term is ``Gamma^T S`` and nothing else. The reference writes this
    slot as ``phi_i,K M_iL``, which is the same object but reached through the
    gradient of the rotation *vector*; that route needs the tangent map ``H``
    and therefore inherits the ``H`` versus ``H^T`` question that
    ``core/rotation.py`` records. Working in ``Gamma`` and ``S`` avoids it.

    Two traps carried over from :func:`eshelby_stress`, and one new
    -------------------------------------------------------------------
    ``Sigma`` is **not symmetric**, so the contraction index is fixed:
    ``Sigma_KL N_L``. It is **invariant** under a superposed spatial rotation
    rather than conjugated by it -- ``E``, ``Gamma``, ``B`` and ``S`` are all
    material-frame and unaffected, and the ``R`` in ``P = Q B`` cancels against
    ``F^T``. And ``P`` must be ``dW/dF`` for the *same* ``W``: here that is
    ``P = Q B``, not ``J sigma F^-T``.

    The new one: unlike the classical case the couple term is **not spherical**,
    so it contributes to ``skew(Sigma)`` and hence to
    :func:`configurational_torque`. That is the whole point of computing it --
    it is the part of the torque a classical continuum cannot produce.
    """
    energy = np.asarray(energy, dtype=float)
    if kinetic is not None:
        energy = energy - np.asarray(kinetic, dtype=float)
    d = F.shape[-1]
    return (energy[:, None, None] * np.eye(d)
            - np.einsum("piK,piL->pKL", F, P)
            - np.einsum("pgK,pgL->pKL", wryness, couple))


def configurational_torque(sigma: np.ndarray) -> np.ndarray:
    r"""Axial vector of ``skew(Sigma)``: the torque *on the material manifold*.

    ``(n, 3, 3) -> (n, 3)`` in 3-D; ``(n, 2, 2) -> (n,)`` in 2-D, where a skew
    tensor has one component.

    Where it comes from
    -------------------
    ``Sigma`` is **not symmetric in general** -- that is already recorded as a
    trap for the contraction index, ``Sigma_KL N_L``. Its antisymmetric part is
    a physical quantity in its own right. Where a material force drives a defect
    to *translate* through the material, this drives it to *rotate*.

    It vanishes identically for any isotropic hyperelastic solid, and the reason
    is worth stating because it makes the zero exact rather than approximate:
    ``F^T P = C S`` with ``S`` an isotropic function of ``C``, so the two are
    coaxial, their product is symmetric, and the skew part is zero at **every**
    ``F``. Anisotropy breaks the coaxiality and the torque appears. So a nonzero
    reading on an isotropic material is a bug, not a small effect.

    Numerically better behaved than the material force
    --------------------------------------------------
    ``Sigma = (W - K) I - F^T P`` carries the energy on its *diagonal*, and a
    spherical tensor has no antisymmetric part, so ``W`` **cancels out of this
    quantity exactly**:

        skew(Sigma) = -skew(F^T P)

    That matters. The material force suffers a near-total cancellation between
    ``W I`` and ``F^T P`` -- near a crack tip ``sigma`` is ~1e5 times ``J``, and
    what survives can be quadrature error the size of the answer, which is why
    ``eshelby_stress_small_strain`` exists. The torque has no such cancellation
    to suffer: the large, nearly-cancelling term is not in it at all.

    Passing the *micropolar* Eshelby stress
    ``Sigma_KL = W d_KL - F_iK P_iL - phi_i,K M_iL`` is the intended use, and
    there the third term is genuinely non-spherical and does contribute.
    """
    sigma = np.asarray(sigma, dtype=float)
    d = sigma.shape[-1]
    if d == 2:
        return 0.5 * (sigma[:, 1, 0] - sigma[:, 0, 1])
    if d != 3:
        raise ValueError(f"expected 2-D or 3-D tensors, got {d}-D")
    return 0.5 * np.stack([sigma[:, 2, 1] - sigma[:, 1, 2],
                           sigma[:, 0, 2] - sigma[:, 2, 0],
                           sigma[:, 1, 0] - sigma[:, 0, 1]], axis=1)


def pseudomomentum(rho0, F: np.ndarray, v: np.ndarray) -> np.ndarray:
    r"""``\mathcal{P} = -rho_0 F^T v`` per unit reference volume.

    ``(n, d, d), (n, d) -> (n, d)``. Note this is a vector on the *material*
    manifold: its index runs over reference coordinates, so it cannot be added
    to a spatial momentum even though both have the same shape.
    """
    rho0 = np.asarray(rho0, dtype=float)
    out = -np.einsum("pji,pj->pi", np.asarray(F), np.asarray(v))
    return (rho0[:, None] if rho0.ndim else rho0) * out


def total_pseudomomentum(mass: np.ndarray, F: np.ndarray,
                         v: np.ndarray) -> np.ndarray:
    r"""``\mathcal{P}_h = -\sum_p m_p F_p^T v_p``. ``-> (d,)``.

    The discrete momentum map of the material-translation group, and the
    quantity a structure-preserving transfer would conserve. Distinct from the
    spatial momentum ``\sum_p m_p v_p``: a scheme can conserve one and not the
    other, which is the point of measuring it.
    """
    mass = np.asarray(mass, dtype=float)
    return -np.einsum("p,pji,pj->i", mass, np.asarray(F), np.asarray(v))


def material_force(sigma: np.ndarray, volume: np.ndarray,
                   grad_w: np.ndarray) -> np.ndarray:
    r"""Contributions to the nodal material force
    ``G_i = \sum_p V_p^0 \Sigma_p \operatorname{Grad} w_{ip}``.

    Parameters
    ----------
    sigma  : ``(n, d, d)`` Eshelby stress at each quadrature point / particle
    volume : ``(n,)`` reference volume carried by each
    grad_w : ``(n, k, d)`` gradients of the ``k`` basis functions in range,
             **with respect to the reference coordinates**

    Returns
    -------
    ``(n, k, d)`` -- one vector per point per node, *unscattered*.

    Not scattered, deliberately: MPM accumulates these by flat node index and
    FEM by element connectivity, and mirroring
    ``ExplicitMPMStepper.internal_force`` keeps the two assemblies the same
    shape. Scatter with ``np.add.at(G, stencil.node_id, contributions)``.

    On ``grad_w``
    -------------
    These must be *material* gradients. In a total-Lagrangian scheme the basis
    is evaluated at fixed reference positions and that is what you have. In an
    updated-Lagrangian scheme the stencil gradients are spatial, and using them
    here computes a different object -- pull them back with ``F^T`` first.
    """
    return np.einsum("p,pij,pkj->pki", np.asarray(volume, dtype=float),
                     np.asarray(sigma), np.asarray(grad_w))


def j_integral(sigma: np.ndarray, volume: np.ndarray,
               grad_q: np.ndarray) -> np.ndarray:
    r"""Rice's J-integral in domain form. ``-> (d,)``, one component per axis.

    .. math::
        J_K = -\int_A \Sigma_{KL}\,\frac{\partial q}{\partial X_L}\,dA

    Parameters
    ----------
    sigma  : ``(n, d, d)`` Eshelby stress at each quadrature point
    volume : ``(n,)`` quadrature weight (area or volume) of each point
    grad_q : ``(n, d)`` gradient of the weight function ``q``, which must be 1
             on a contour enclosing the tip, 0 on an outer contour, and smooth
             between. Only the ring where ``grad_q`` is nonzero contributes, so
             only that ring has to be sampled.

    The domain form rather than the contour form, because it needs no topology:
    given ``Sigma`` and ``grad q`` at points it is a contraction, so the same
    function serves FEM, MPM and an analytical field. Extracting a contour from
    a mesh is the part that would not generalise.

    Which Eshelby stress
    --------------------
    For linear elasticity use :func:`eshelby_stress_small_strain`. Passing the
    finite-strain :func:`eshelby_stress` is not wrong but converges far more
    slowly -- see the note there.

    The trap that produces a plausible wrong answer
    -----------------------------------------------
    ``P`` must be ``dW/dF`` for the *same* ``W`` passed as the energy. For a
    linear-elastic ``W = eps:C:eps/2`` that means ``P = sigma`` exactly; forming
    ``P = J sigma F^-T`` instead -- the correct Cauchy-to-Piola relation, but for
    a different constitutive statement -- converges cleanly to **half** the true
    ``J``, which looks like a factor-of-two convention error and is not.
    """
    return -np.einsum("p,pkj,pj->k", np.asarray(volume, dtype=float),
                      np.asarray(sigma), np.asarray(grad_q))
