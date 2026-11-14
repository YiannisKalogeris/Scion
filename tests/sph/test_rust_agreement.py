"""
The Rust crate against the NumPy path, on the whole example.

The NumPy path is the reference and the crate is what has to agree with it,
as for the MPM kernels. Unlike those, **bit identity is not claimed here**:
the two accumulate the same pair sums in different orders -- the cell list
walks offsets in Python and particles in Rust -- so the last bits differ by
construction. What is asserted is agreement at round-off, which is what an
order-of-summation difference can produce and a wrong formula cannot.

Skipped when cargo is absent, because the library must stay usable with no
compiler.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from examples.sph.heat_transfer_2d import run

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(shutil.which("cargo") is None,
                                reason="cargo is not installed")


def _run_rust(tmp_path: Path) -> tuple[np.ndarray, np.ndarray]:
    out = tmp_path / "heat.csv"
    result = subprocess.run(
        ["cargo", "run", "-p", "scion-sph", "--release", "--quiet",
         "--example", "heat_transfer_2d", "--", str(out)],
        cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"cargo failed:\n{result.stdout}\n{result.stderr}")
    history = np.genfromtxt(out, delimiter=",", skip_header=1)
    field = np.genfromtxt(tmp_path / "heat_field.csv", delimiter=",", skip_header=1)
    return history, field


def test_observers_agree_at_round_off(tmp_path):
    rust_history, _ = _run_rust(tmp_path)
    _, _, history = run(quiet=True)
    python_history = np.array([[t, *row] for t, row in history])
    assert rust_history.shape == python_history.shape
    resolved = np.abs(python_history) > 1e-12
    relative = (np.abs(rust_history - python_history)[resolved]
                / np.abs(python_history)[resolved])
    assert relative.max() < 1e-13, f"worst relative gap {relative.max():.2e}"


def test_the_whole_field_agrees_at_round_off(tmp_path):
    """Eleven observers are eleven kernel averages and could hide a local
    disagreement; this is all 8000 particles."""
    _, rust_field = _run_rust(tmp_path)
    solver, phi, _ = run(quiet=True)
    assert rust_field.shape[0] == len(phi)
    assert np.abs(rust_field[:, :2] - solver.positions).max() < 1e-15
    gap = np.abs(rust_field[:, 2] - phi)
    assert gap.max() < 1e-15 * max(np.abs(phi).max(), 1.0)


def test_taylor_green_agrees_with_numpy(tmp_path):
    """The fluid path, which has a great deal more state than the diffusion
    one: moving particles, a rebuilt neighbour list every advection step, two
    half-steps and a transport shift. Agreement here is a much stronger
    statement about the port than the diffusion case was."""
    out = tmp_path / "tg.csv"
    result = subprocess.run(
        ["cargo", "run", "-p", "scion-sph", "--release", "--quiet",
         "--example", "taylor_green_2d", "--", "0.2", "60", str(out)],
        cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"cargo failed:\n{result.stdout}\n{result.stderr}")
    rust = np.genfromtxt(out, delimiter=",", skip_header=1)

    from examples.sph.taylor_green_2d import run as run_taylor_green
    _, history, _, _ = run_taylor_green(n=60, end_time=0.2, interval=0.1, quiet=True)
    python = np.array([[t, energy, speed] for t, energy, speed, _ in history])

    assert rust.shape == python.shape
    relative = np.abs(rust - python) / np.maximum(np.abs(python), 1e-30)
    # NOT 1e-15 as in the diffusion case, and the difference is physical
    # rather than sloppy: those particles never move, so a round-off
    # difference in the pair sums stays round-off. Here the particles advect,
    # so the two runs' trajectories separate exponentially from a 1e-16 seed
    # -- measured 3e-10 after about two hundred acoustic steps, growing with
    # the run length. Tightening this threshold would make the test a
    # stopwatch on chaos.
    assert relative.max() < 1e-7, f"worst relative gap {relative.max():.2e}"
    first = np.abs(rust[1] - python[1]) / np.abs(python[1])
    assert first.max() < relative.max(), "the gap should grow with time, not shrink"


def test_dam_break_confinement_agrees_with_numpy(tmp_path):
    """The level-set boundary as well as the fluid: the kernel integrals,
    the cut-cell fraction, the surface bounding and the free-surface clamp
    all have to match, on a flow that touches a wall and an obstacle."""
    out = tmp_path / "dam.csv"
    result = subprocess.run(
        ["cargo", "run", "-p", "scion-sph", "--release", "--quiet",
         "--example", "dam_break_confinement", "--", "0.4", str(out)],
        cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"cargo failed:\n{result.stdout}\n{result.stderr}")
    rust = np.genfromtxt(out, delimiter=",", skip_header=1)

    from examples.sph.dam_break_confinement import run as run_dam_break
    _, history = run_dam_break(end_time=0.4, interval=0.1, quiet=True)
    python = np.array([[t, energy, speed, out_count]
                       for t, energy, speed, _front, out_count in history])

    assert rust.shape == python.shape
    relative = np.abs(rust - python) / np.maximum(np.abs(python), 1e-30)
    assert relative.max() < 1e-9, f"worst relative gap {relative.max():.2e}"


def test_oscillating_beam_agrees_with_numpy(tmp_path):
    """The solid path: a fixed reference neighbour list, an integrated
    deformation gradient and a clamp. Nothing is rebuilt, so unlike the
    fluid cases the two runs do not separate -- the agreement should hold at
    round-off for as long as it is run."""
    out = tmp_path / "beam.csv"
    result = subprocess.run(
        ["cargo", "run", "-p", "scion-sph", "--release", "--quiet",
         "--example", "oscillating_beam", "--", "0.05", str(out)],
        cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"cargo failed:\n{result.stdout}\n{result.stderr}")
    rust = np.genfromtxt(out, delimiter=",", skip_header=1)

    from examples.sph.oscillating_beam import run as run_beam
    _, history = run_beam(end_time=0.05, interval=0.005, quiet=True)
    python = np.array(history)
    assert rust.shape == python.shape
    resolved = np.abs(python) > 1e-12
    relative = np.abs(rust - python)[resolved] / np.abs(python)[resolved]
    assert relative.max() < 1e-10, f"worst relative gap {relative.max():.2e}"
