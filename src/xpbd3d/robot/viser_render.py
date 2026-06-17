"""Render a :class:`RobotModel` in a viser scene and animate it from FK.

Layout: one ``add_frame`` per link at its world pose, with the link's visual
meshes and collision primitives parented underneath (each at its geom-local
pose). Because viser scene nodes are hierarchical by name, moving a joint only
needs the affected link **frames** updated — the geometry rides along, so
``update(q)`` never re-uploads meshes.

Visuals and collisions are independent node sets with their own visibility, so
the viewer can toggle each on/off (collision is drawn as a wireframe overlay).
"""

from __future__ import annotations

import numpy as np
import trimesh.transformations as tf

from . import transforms as T
from .model import RobotModel

_COLLISION_RGB = (255, 140, 30)


def _u8(c):
    return (int(np.clip(c[0], 0, 1) * 255), int(np.clip(c[1], 0, 1) * 255),
            int(np.clip(c[2], 0, 1) * 255))


def _decompose(M: np.ndarray):
    """4x4 → (position xyz, quaternion wxyz)."""
    return tuple(M[:3, 3]), tuple(tf.quaternion_from_matrix(M))


class RobotView:
    def __init__(self, server, model: RobotModel, prefix: str = "/robot",
                 lift: bool = True, clearance: float = 0.02,
                 show_visual: bool = True, show_collision: bool = False):
        self.server = server
        self.model = model
        self.prefix = prefix
        self.base_offset = T.identity()
        self._frames: dict = {}
        self._visual_nodes: list = []
        self._collision_nodes: list = []

        # Pre-load every geometry mesh once, keyed by (link, kind, index).
        self._items = []  # (node_name, link_name, gi, mesh, is_collision)
        for lname, link in model.links.items():
            for i, gi in enumerate(link.visuals):
                m = gi.geometry.to_trimesh()
                if m is not None:
                    self._items.append((f"{prefix}/{lname}/vis{i}", lname, gi, m, False))
            for i, gi in enumerate(link.collisions):
                m = gi.geometry.to_trimesh()
                if m is not None:
                    self._items.append((f"{prefix}/{lname}/col{i}", lname, gi, m, True))

        if lift:
            self.base_offset = self._auto_lift(clearance)

        self._build()
        self.set_visual_visible(show_visual)
        self.set_collision_visible(show_collision)

    # -- world transforms (FK with the standing/base offset folded in) --------
    def _world(self, q=None) -> dict:
        fk = self.model.forward_kinematics(q)
        if not np.allclose(self.base_offset, np.eye(4)):
            fk = {k: self.base_offset @ M for k, M in fk.items()}
        return fk

    def _auto_lift(self, clearance: float) -> np.ndarray:
        """Translate the whole robot in +z so its lowest vertex (at the default
        pose) sits ``clearance`` above the floor — makes it stand on the grid
        regardless of the source format's base height."""
        fk = self.model.forward_kinematics()
        min_z = np.inf
        for _, lname, gi, mesh, _ in self._items:
            W = fk[lname] @ gi.origin
            v = np.asarray(mesh.vertices, np.float64)
            zs = (v @ W[:3, :3].T + W[:3, 3])[:, 2]
            min_z = min(min_z, float(zs.min()))
        if not np.isfinite(min_z):
            return T.identity()
        return T.translation((0.0, 0.0, clearance - min_z))

    # -- build / update -------------------------------------------------------
    def _build(self):
        fk = self._world()
        for name in self.model.links:
            pos, wxyz = _decompose(fk[name])
            self._frames[name] = self.server.scene.add_frame(
                f"{self.prefix}/{name}", show_axes=False, position=pos, wxyz=wxyz)
        for node_name, _, gi, mesh, is_col in self._items:
            pos, wxyz = _decompose(gi.origin)
            verts = np.asarray(mesh.vertices, np.float32)
            faces = np.asarray(mesh.faces, np.uint32)
            if is_col:
                h = self.server.scene.add_mesh_simple(
                    node_name, verts, faces, color=_COLLISION_RGB, wireframe=True,
                    opacity=0.6, side="double", position=pos, wxyz=wxyz)
                self._collision_nodes.append(h)
            else:
                h = self.server.scene.add_mesh_simple(
                    node_name, verts, faces, color=_u8(gi.color), flat_shading=False,
                    side="double", position=pos, wxyz=wxyz)
                self._visual_nodes.append(h)

    def update(self, q: dict | None):
        fk = self._world(q)
        self.update_world(fk)

    def update_world(self, world: dict):
        """Place link frames directly from a ``{link_name: 4x4}`` map (e.g. poses
        derived from a physics solver), bypassing forward kinematics."""
        with self.server.atomic():
            for name, fr in self._frames.items():
                M = world.get(name)
                if M is None:
                    continue
                pos, wxyz = _decompose(M)
                fr.position = pos
                fr.wxyz = wxyz

    def set_visual_visible(self, on: bool):
        for h in self._visual_nodes:
            h.visible = bool(on)

    def set_collision_visible(self, on: bool):
        for h in self._collision_nodes:
            h.visible = bool(on)

    @property
    def n_visual(self) -> int:
        return len(self._visual_nodes)

    @property
    def n_collision(self) -> int:
        return len(self._collision_nodes)
