"""Numerical examples for micropolar phase-field fracture on MPM.

Two problems, chosen as a contrast rather than as two demonstrations of the
same thing:

``notched_tension``
    A notched plate in tension. Stretch-dominated, so the microrotation field
    is nearly uniform, the wryness is small and the internal length has almost
    no route into the damage evolution. The deliberate *null*.

``notched_beam``
    A notched cantilever a few internal lengths deep, loaded in bending.
    Bending is the mode that produces a systematic rotation gradient through
    the depth, so this is where the internal length reaches the crack.

Both sweep ``l_b`` with the geometry, the mesh, the phase-field length and
``G_c`` held fixed, so a classical model -- which has no length of its own --
would return an identical answer for every row of the sweep.
"""
