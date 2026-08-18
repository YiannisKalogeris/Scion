"""
Warp backend for MPM.

Import is deliberately NOT re-exported from the parent ``mpm`` package: that
would make ``import scion.discretization.mpm`` pull in ``warp`` on every
machine, including ones that do not have it. Import this subpackage explicitly:

    from scion.discretization.mpm.warp import WarpMPMStepper

Install with ``pip install warp-lang``.
"""

from .device import device_info, get_device, sync
from .stepper import WarpMPMStepper, WarpParticleSet

__all__ = [
    "get_device",
    "device_info",
    "sync",
    "WarpParticleSet",
    "WarpMPMStepper",
]
