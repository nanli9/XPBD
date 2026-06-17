"""Solver-agnostic robot model: links, joints, geometry and forward kinematics.

Both :func:`xpbd3d.robot.load_urdf` and :func:`xpbd3d.robot.load_mjcf` parse into
this one representation, so the viewer (and, later, the physics solvers) only
need to understand a single data model. It depends only on ``numpy`` +
``trimesh`` — no Warp, no solver — so the same model can feed XPBD and AVBD.

Conventions: every transform is a 4x4 homogeneous ``float64`` matrix; lengths
are in metres; both URDF and MJCF robots are **Z-up**. Geometry sizes are stored
*normalised* — ``box_size`` is the full extent, ``length`` the full length —
regardless of the source format's half/full convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from . import transforms as T

# Joint types (kinematic only; "floating"/"free" carry 6 DOF we leave at rest).
FIXED = "fixed"
REVOLUTE = "revolute"
CONTINUOUS = "continuous"
PRISMATIC = "prismatic"
FLOATING = "floating"

_ACTUATED = (REVOLUTE, CONTINUOUS, PRISMATIC)


@dataclass
class Geometry:
    """A single visual or collision primitive.

    ``kind`` ∈ {mesh, box, cylinder, sphere, capsule, ellipsoid}. Sizes are
    full-extent (not half): a box of ``box_size=(0.2,0.1,0.1)`` is 20×10×10 cm.
    """

    kind: str
    mesh_path: str | None = None
    scale: tuple = (1.0, 1.0, 1.0)
    box_size: tuple = (1.0, 1.0, 1.0)
    radius: float = 0.1
    length: float = 0.1  # full length (cylinder/capsule)

    def to_trimesh(self) -> trimesh.Trimesh | None:
        """Build a :class:`trimesh.Trimesh` for rendering, or ``None`` if a mesh
        file could not be loaded (caller may fall back to collision shapes)."""
        try:
            if self.kind == "mesh":
                if not self.mesh_path:
                    return None
                m = trimesh.load(self.mesh_path, force="mesh", process=False)
                if m is None or not hasattr(m, "vertices") or len(m.vertices) == 0:
                    return None
                if tuple(self.scale) != (1.0, 1.0, 1.0):
                    m = m.copy()
                    m.apply_scale(np.asarray(self.scale, dtype=np.float64))
                return m
            if self.kind == "box":
                return trimesh.creation.box(extents=np.asarray(self.box_size, np.float64))
            if self.kind == "sphere":
                return trimesh.creation.icosphere(subdivisions=2, radius=float(self.radius))
            if self.kind == "cylinder":
                return trimesh.creation.cylinder(radius=float(self.radius),
                                                 height=float(self.length), sections=24)
            if self.kind == "capsule":
                return trimesh.creation.capsule(radius=float(self.radius),
                                                height=float(self.length), count=[12, 12])
            if self.kind == "ellipsoid":
                m = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
                m.apply_scale(np.asarray(self.box_size, np.float64) * 0.5)
                return m
        except Exception as exc:  # pragma: no cover - mesh I/O is best-effort
            print(f"[robot] geometry '{self.kind}' ({self.mesh_path}): {exc}")
        return None


@dataclass
class GeomInstance:
    """A geometry placed in its link frame, with a render colour."""

    origin: np.ndarray  # 4x4, geometry pose relative to the owning link
    geometry: Geometry
    color: tuple = (0.7, 0.7, 0.75)


@dataclass
class Link:
    name: str
    visuals: list = field(default_factory=list)      # list[GeomInstance]
    collisions: list = field(default_factory=list)   # list[GeomInstance]
    mass: float = 0.0                                 # from <inertial> (0 = unknown)


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray                 # 4x4, child frame relative to parent at q=0
    axis: tuple = (1.0, 0.0, 0.0)      # in the child/joint frame
    lower: float = 0.0
    upper: float = 0.0
    anchor: tuple = (0.0, 0.0, 0.0)    # joint position in the child frame (MJCF; URDF=0)

    @property
    def actuated(self) -> bool:
        return self.type in _ACTUATED


class RobotModel:
    """A parsed robot: links keyed by name, a joint list, and a kinematic tree."""

    def __init__(self, name: str, links: dict, joints: list,
                 root: str, root_origin: np.ndarray | None = None):
        self.name = name
        self.links: dict[str, Link] = links
        self.joints: list[Joint] = joints
        self.root = root
        self.root_origin = T.identity() if root_origin is None else root_origin
        # child-link -> joint, and parent -> [joints], for tree walks
        self._by_child = {j.child: j for j in joints}
        self._children: dict[str, list] = {}
        for j in joints:
            self._children.setdefault(j.parent, []).append(j)

    @property
    def actuated_joints(self) -> list:
        return [j for j in self.joints if j.actuated]

    def default_q(self) -> dict:
        """Zero (or limit-clamped-to-zero) angle for every actuated joint."""
        q = {}
        for j in self.actuated_joints:
            lo, hi = j.lower, j.upper
            q[j.name] = 0.0 if lo <= 0.0 <= hi or lo == hi else 0.5 * (lo + hi)
        return q

    def joint_motion(self, joint: Joint, q: float) -> np.ndarray:
        if joint.type in (REVOLUTE, CONTINUOUS):
            rot = T.axis_angle_motion(joint.axis, q)
            if tuple(joint.anchor) == (0.0, 0.0, 0.0):
                return rot
            # Rotate about the line through ``anchor``: T(a) · R · T(-a).
            return T.translation(joint.anchor) @ rot @ T.translation(-np.asarray(joint.anchor))
        if joint.type == PRISMATIC:
            return T.prismatic_motion(joint.axis, q)
        return T.identity()

    def forward_kinematics(self, q: dict | None = None) -> dict:
        """World 4x4 transform for every link, given actuated-joint values ``q``
        (radians / metres). Missing joints default to 0."""
        q = {} if q is None else q
        world = {self.root: self.root_origin.copy()}
        # Walk parents-before-children: repeatedly resolve links whose parent is known.
        pending = list(self.links.keys())
        guard = 0
        while pending and guard <= len(self.links) + 1:
            guard += 1
            still = []
            for name in pending:
                if name in world:
                    continue
                j = self._by_child.get(name)
                if j is None or j.parent not in world:
                    still.append(name)
                    continue
                motion = self.joint_motion(j, float(q.get(j.name, 0.0)))
                world[name] = world[j.parent] @ j.origin @ motion
            pending = still
        # Any leftover (disconnected) links: drop at the root frame.
        for name in pending:
            world[name] = self.root_origin.copy()
        return world

    def __repr__(self) -> str:
        return (f"RobotModel(name={self.name!r}, links={len(self.links)}, "
                f"joints={len(self.joints)}, actuated={len(self.actuated_joints)}, "
                f"root={self.root!r})")
