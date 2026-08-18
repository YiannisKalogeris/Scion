"""
Particle-cloud visualisation with PyVista.

Companion to ``viewer.py``, which plots meshes. Kept separate because a particle
cloud is a ``pv.PolyData`` of bare points with no cells, so none of the
``_to_pyvista`` mesh machinery applies.

For anything beyond a quick look, prefer the file-based route: write ``.vtu``
with ``scion.io.particles`` and open the series in ParaView or OVITO.
These helpers are for the development loop and for automated figures.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["to_pyvista_points", "plot_particles", "save_particle_frames"]


def to_pyvista_points(positions: np.ndarray,
                      point_data: dict[str, np.ndarray] | None = None) -> Any:
    """Convert particle positions (+ fields) to a PyVista PolyData."""
    import pyvista as pv

    pts = np.asarray(positions, dtype=np.float64)
    if pts.shape[1] < 3:
        pts = np.hstack([pts, np.zeros((pts.shape[0], 3 - pts.shape[1]))])

    cloud = pv.PolyData(pts)
    for name, arr in (point_data or {}).items():
        cloud.point_data[name] = np.asarray(arr)
    return cloud


def plot_particles(
    positions: np.ndarray,
    scalars: np.ndarray | None = None,
    scalar_name: str = "field",
    radius: float | None = None,
    cmap: str = "viridis",
    off_screen: bool = False,
    screenshot: str | None = None,
    **kwargs: Any,
) -> None:
    """Plot a particle cloud.

    Parameters
    ----------
    positions   : (n, 2) or (n, 3)
    scalars     : (n,) or (n, dim); vectors are shown as magnitude
    radius      : sphere radius. If given, particles are rendered as spheres
                  via ``render_points_as_spheres``; leave as None for flat points,
                  which is far faster above ~10^5 particles.
    off_screen  : render without opening a window (for scripted figures)
    screenshot  : path to save a PNG; implies ``off_screen``
    """
    import pyvista as pv

    data = None
    if scalars is not None:
        s = np.asarray(scalars)
        data = {scalar_name: s if s.ndim == 1 else np.linalg.norm(s, axis=1)}

    cloud = to_pyvista_points(positions, data)

    plotter = pv.Plotter(off_screen=off_screen or screenshot is not None)
    plotter.add_mesh(
        cloud,
        scalars=scalar_name if data else None,
        cmap=cmap,
        render_points_as_spheres=radius is not None,
        point_size=(radius * 1000 if radius is not None else 5.0),
        **kwargs,
    )
    plotter.view_xy() if np.asarray(positions).shape[1] == 2 else None
    if screenshot is not None:
        plotter.screenshot(screenshot)
        plotter.close()
    else:
        plotter.show()


def save_particle_frames(
    series_positions,
    series_scalars=None,
    directory: str = "frames",
    scalar_name: str = "field",
    clim: tuple[float, float] | None = None,
    **kwargs: Any,
) -> list[str]:
    """Render a list of frames to PNGs with a fixed colour range.

    Fixing ``clim`` across frames matters: PyVista rescales per frame by
    default, which makes an animation look like the field is oscillating when
    it is not.
    """
    import pyvista as pv

    from pathlib import Path

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    if clim is None and series_scalars is not None:
        lo = min(float(np.min(s)) for s in series_scalars)
        hi = max(float(np.max(s)) for s in series_scalars)
        clim = (lo, hi)

    for i, pos in enumerate(series_positions):
        scal = None if series_scalars is None else series_scalars[i]
        path = out / f"frame_{i:06d}.png"
        plot_particles(pos, scal, scalar_name=scalar_name,
                       screenshot=str(path), clim=clim, **kwargs)
        paths.append(str(path))
    return paths
