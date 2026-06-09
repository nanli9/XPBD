"""xpbd3d — 3D Extended Position-Based Dynamics in NVIDIA Warp.

Macklin, Müller, Chentanez. "XPBD: Position-Based Simulation of Compliant
Constrained Dynamics." MiG 2016 (see ``reference/XPBD_Macklin2016.pdf``).
"""

from .scene import Body, ConstraintHandle, Shape
from .solver import ATTACH, CONTACT, DISTANCE, FLOOR, Solver
from .solver_6dof import RigidBody, Solver6DOF

__all__ = [
    "Solver",
    "Solver6DOF",
    "RigidBody",
    "Body",
    "Shape",
    "ConstraintHandle",
    "DISTANCE",
    "ATTACH",
    "FLOOR",
    "CONTACT",
]
