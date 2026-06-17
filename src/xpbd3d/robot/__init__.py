"""Robot-description loaders (URDF + MJCF) → a common :class:`RobotModel`.

Solver-agnostic: depends only on ``numpy`` + ``trimesh``, so the same parsed
model can be visualised (``viser_render``) and, later, driven by the XPBD or
AVBD solvers without re-parsing.

    from xpbd3d.robot import load_mjcf, load_urdf
    model = load_mjcf("/path/to/go2.xml")
    model = load_urdf("/path/to/go2_description.urdf")
"""

from __future__ import annotations

from .model import Geometry, GeomInstance, Joint, Link, RobotModel
from .mjcf import load_mjcf
from .urdf import load_urdf

__all__ = [
    "RobotModel",
    "Link",
    "Joint",
    "Geometry",
    "GeomInstance",
    "load_urdf",
    "load_mjcf",
]
