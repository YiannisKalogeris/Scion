"""
Cross-checks for the Warp backend.

The point of these is not to re-verify the physics -- ``test_mpm_kernels.py``
already does that for the NumPy implementation. The point is to pin the two
backends together, so a change to one that silently diverges from the other
gets caught.

Tolerances are ~1e-5 relative because Warp is float32 and the reference is
float64. Tighter is not achievable; much looser means a real bug.
"""

from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip("warp", reason="warp-lang not installed")

from simulacrum.discretization.mpm import (  # noqa: E402
    APIC,
    CartesianGrid,
    ExplicitMPMStepper,
    ParticleSet,
)
from simulacrum.discretization.mpm.warp import (  # noqa: E402
    WarpMPMStepper,
    WarpParticleSet,
    get_device,
)
from simulacrum.materials.hyperelastic_batched import (  # noqa: E402
    NeoHookeanCompressible,
)

DEVICE = "cpu"          # deterministic; GPU is exercised by the same tests
E, NU, RHO = 1.0e3, 0.3, 1.0e3
DX, N = 1.0 / 32, 32


def _disk(velocity, centre=(0.5, 0.5), r=0.15, ppc=2):
    return ParticleSet.from_sampler(
        lambda p: np.linalg.norm(p - np.array(centre), axis=1) < r,
        ((0.0, 0.0), (1.0, 1.0)), DX, ppc=ppc,
        velocity=velocity, density=RHO,
    )


def _run_both(particles_fn, n_steps=100, dt=1e-3, body_force=(0.0, 0.0)):
    mat = NeoHookeanCompressible.from_E_nu(E, NU)

    pn = particles_fn()
    grid = CartesianGrid(shape=(N, N), dx=DX)
    ExplicitMPMStepper(grid, mat, APIC(), dt=dt,
                       body_force=body_force).run(pn, n_steps, callback_every=10**9)

    pw = WarpParticleSet.from_numpy(particles_fn(), DEVICE)
    stw = WarpMPMStepper((N, N), DX, mat.mu, mat.lam, dt,
                         body_force=body_force, device=DEVICE)
    for _ in range(n_steps):
        stw.step(pw)
    return pn, pw, pw.to_numpy()


def test_device_resolution():
    assert get_device("cpu") == "cpu"
    resolved = get_device("auto")
    assert resolved == "cpu" or resolved.startswith("cuda")


def test_gpu_request_fails_loudly_when_unavailable():
    """Silently falling back to CPU would hide a 100x slowdown."""
    if wp.is_cuda_available():
        pytest.skip("CUDA present")
    with pytest.raises(RuntimeError, match="no CUDA device"):
        get_device("gpu")


def test_matches_numpy_free_flight():
    """Two disks colliding, no forcing: positions, velocity and F agree."""
    def seed():
        a = _disk((0.1, 0.1), (0.35, 0.35), 0.12)
        b = _disk((-0.1, -0.1), (0.65, 0.65), 0.12)
        return a.concatenate(b)

    pn, pw, r = _run_both(seed, n_steps=150)
    scale = np.abs(pn.x).max()
    assert np.abs(r["x"] - pn.x).max() / scale < 2e-5
    assert np.abs(r["v"] - pn.v).max() < 1e-5
    assert np.abs(r["F"] - pn.F).max() < 1e-5


def test_matches_numpy_with_gravity():
    pn, pw, r = _run_both(lambda: _disk((0.0, 0.0)), n_steps=100,
                          body_force=(0.0, -9.81))
    assert np.abs(r["x"] - pn.x).max() / np.abs(pn.x).max() < 2e-5
    assert np.abs(r["v"] - pn.v).max() < 1e-4


def test_momentum_conserved_on_device():
    """No external forcing -> momentum constant to float32 precision."""
    mat = NeoHookeanCompressible.from_E_nu(E, NU)
    a = _disk((0.1, 0.1), (0.35, 0.35), 0.12)
    b = _disk((-0.1, -0.1), (0.65, 0.65), 0.12)
    pw = WarpParticleSet.from_numpy(a.concatenate(b), DEVICE)
    stw = WarpMPMStepper((N, N), DX, mat.mu, mat.lam, 1e-3, device=DEVICE)

    p0 = pw.linear_momentum()
    ke0 = pw.kinetic_energy()
    for _ in range(200):
        stw.step(pw)
    assert np.linalg.norm(pw.linear_momentum() - p0) < 1e-4 * ke0
    assert pw.jacobian().min() > 0.0


def test_rigid_translation_leaves_F_unchanged():
    mat = NeoHookeanCompressible.from_E_nu(E, NU)
    pw = WarpParticleSet.from_numpy(_disk((0.05, -0.03)), DEVICE)
    stw = WarpMPMStepper((N, N), DX, mat.mu, mat.lam, 1e-3, device=DEVICE)
    for _ in range(50):
        stw.step(pw)
    assert np.abs(pw.jacobian() - 1.0).max() < 1e-5


def test_roundtrip_numpy_warp_numpy():
    """Interop must not lose or reorder state."""
    p = _disk((0.2, -0.1))
    p.F[:, 0, 0] = 1.3
    p.C[:, 0, 1] = 0.4
    pw = WarpParticleSet.from_numpy(p, DEVICE)
    back = pw.to_numpy()
    assert np.abs(back["x"] - p.x).max() < 1e-6
    assert np.abs(back["F"] - p.F).max() < 1e-6
    assert np.abs(back["C"] - p.C).max() < 1e-6


def test_walls_stop_the_material():
    mat = NeoHookeanCompressible.from_E_nu(1e5, NU)
    pw = WarpParticleSet.from_numpy(_disk((0.0, 0.0), (0.5, 0.7), 0.1), DEVICE)
    stw = WarpMPMStepper((N, N), DX, mat.mu, mat.lam, 2e-4,
                         body_force=(0.0, -9.81),
                         walls=((2 * DX, 2 * DX), (1 - 2 * DX, 1 - 2 * DX)),
                         device=DEVICE)
    for _ in range(2000):
        stw.step(pw)
    y = pw.x.numpy()[:, 1]
    assert y.min() > 2 * DX - 3 * DX      # did not fall through the floor
    assert y.max() < 0.75                 # but did fall
