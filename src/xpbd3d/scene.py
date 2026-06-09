"""Light-weight scene-building handles.

The actual simulation state lives as flat Warp arrays inside ``Solver``; these
objects just remember indices and carry render metadata. Mirrors the structure
of the AVBD 3D reference so the two solvers' viewers feel the same.

``Body.shape`` is metadata the viewer reads — the XPBD solver treats every body
as a 3-DOF point mass (a single particle). Rigid 6-DOF bodies are a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Shape:
    """Visual primitive attached to a body. The solver treats the body as a
    point mass; this only affects how the viewer draws it (and, for ``sphere``,
    the collision radius used by the broad phase)."""

    kind: Literal["sphere", "cube", "pillar"] = "sphere"
    # sphere: (radius,). cube: (hx, hy, hz) half-extents. pillar: (radius, half_height)
    size: tuple[float, ...] = (0.1,)
    color: tuple[float, float, float] = (0.4, 0.6, 0.9)


@dataclass
class Body:
    """Handle to a particle in the solver. ``index`` is the row in the body
    arrays (positions / velocities / inverse-mass)."""

    index: int
    shape: Shape = field(default_factory=Shape)


@dataclass
class ConstraintHandle:
    """Handle to one or more constraint rows in the solver."""

    index: int
    rows: int = 1
