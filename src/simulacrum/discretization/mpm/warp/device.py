"""
Device selection for the Warp backend.

One place decides CPU vs GPU, so nothing else in the backend hardcodes a
device string. Every array allocation and kernel launch takes ``device=...``
from here.

On a Mac, ``warp-lang`` installs and runs but is CPU-only: GPU execution needs
a CUDA-capable NVIDIA device, and there is no Metal backend. That is fine for
development -- CPU Warp still JIT-compiles to native multithreaded code and is
typically one to two orders of magnitude faster than the NumPy reference -- but
the GPU numbers only appear on a CUDA machine. The same source runs on both;
only the string returned here changes.

Precision: Warp's ``vec2``/``mat22`` are float32, which is the normal choice
for GPU MPM. The NumPy reference in the parent package is float64, so expect
agreement to roughly 1e-5 relative, not machine precision. Anything worse than
that is a real discrepancy, not round-off.
"""

from __future__ import annotations

import warp as wp

__all__ = ["get_device", "device_info", "sync"]

_INITIALISED = False


def _ensure_init() -> None:
    global _INITIALISED
    if not _INITIALISED:
        wp.init()
        _INITIALISED = True


def get_device(prefer: str = "auto") -> str:
    """Resolve a device string.

    Parameters
    ----------
    prefer : "auto"  -- CUDA if present, otherwise CPU (the default)
             "cpu"   -- force CPU, useful for bit-comparable debugging
             "gpu"   -- force CUDA, raising if unavailable rather than
                        silently running 100x slower on the CPU

    Any explicit Warp device string ("cuda:1") is passed through.
    """
    _ensure_init()
    if prefer == "cpu":
        return "cpu"
    if prefer in ("gpu", "cuda"):
        if not wp.is_cuda_available():
            raise RuntimeError(
                "GPU requested but no CUDA device is available. On macOS this "
                "is expected: Warp has no Metal backend. Use prefer='cpu' or "
                "run on a CUDA machine."
            )
        return "cuda:0"
    if prefer == "auto":
        return "cuda:0" if wp.is_cuda_available() else "cpu"
    return prefer


def device_info() -> dict:
    """Summary for logging at the start of a run."""
    _ensure_init()
    return {
        "warp_version": wp.config.version,
        "cuda_available": bool(wp.is_cuda_available()),
        "devices": [str(d) for d in wp.get_devices()],
    }


def sync(device: str) -> None:
    """Block until queued kernels finish. No-op on CPU.

    Needed before timing a GPU run, and before reading results back with
    ``.numpy()`` if you have disabled Warp's automatic synchronisation.
    """
    if str(device).startswith("cuda"):
        wp.synchronize_device(device)
