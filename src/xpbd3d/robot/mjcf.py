"""MJCF (MuJoCo XML) loader → :class:`RobotModel`.

A Python port of the renderer's ``MjcfParser.cpp``. The non-trivial parts MuJoCo
models rely on, all handled here:

* ``<include>`` — referenced files are expanded in place (recursively).
* ``<default>`` **class inheritance** + ``childclass`` — geoms/joints inherit a
  parent-chain of attribute defaults; a body's ``childclass`` supplies the class
  for descendants that don't name their own (e.g. ``class="visual"``).
* ``<asset>`` mesh/material maps (a mesh with no ``name`` is keyed by its file
  basename, per MuJoCo).
* ``<worldbody>`` DFS into links + joints (``hinge``→revolute, ``slide``→
  prismatic, ``free``/``<freejoint>``→floating, none→fixed).

MuJoCo sizes are half-extents/half-lengths; we normalise to full extents to
match :class:`Geometry`.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np

from . import transforms as T
from .model import Geometry, GeomInstance, Joint, Link, RobotModel

_DEFAULT_GEOM_RGBA = (0.5, 0.5, 0.5, 1.0)
_COLLISION_COLOR = (0.95, 0.55, 0.2)


def _floats(text, n=None, default=0.0):
    if text is None:
        return [] if n is None else [default] * n
    vals = [float(v) for v in text.replace(",", " ").split()]
    if n is not None and len(vals) < n:
        vals += [default] * (n - len(vals))
    return vals if n is None else vals[:n]


# ---------------------------------------------------------------------------
# Include expansion: flatten the main file + included files into (node, dir) list
# ---------------------------------------------------------------------------
def _collect_nodes(path: str):
    """Return ``[(child_node, base_dir), ...]`` for every top-level child of the
    ``<mujoco>`` root, expanding ``<include>`` recursively. ``base_dir`` is the
    directory of the file that declared the node (for mesh path resolution)."""
    path = os.path.abspath(path)
    base_dir = os.path.dirname(path)
    root = ET.parse(path).getroot()
    out = []
    for child in root:
        if child.tag == "include":
            inc = child.get("file", "")
            inc_path = inc if os.path.isabs(inc) else os.path.join(base_dir, inc)
            out.extend(_collect_nodes(inc_path))
        else:
            out.append((child, base_dir))
    return out


# ---------------------------------------------------------------------------
# Default-class table with parent-chain inheritance
# ---------------------------------------------------------------------------
class _Defaults:
    """``defaults[class_name][element_tag] -> {attr: value}`` resolved through
    the ``<default>`` nesting (each class inherits its parent class)."""

    def __init__(self):
        self.table: dict = {}

    def _process(self, node, parent_cls):
        cls = node.get("class", parent_cls)
        entry = self.table.setdefault(cls, {})
        # Inherit every element default from the parent class first.
        if parent_cls in self.table:
            for tag, attrs in self.table[parent_cls].items():
                merged = dict(attrs)
                merged.update(entry.get(tag, {}))
                entry[tag] = merged
        # Apply this block's own per-element defaults (override inherited).
        for child in node:
            if child.tag == "default":
                continue
            entry.setdefault(child.tag, {}).update(child.attrib)
        # Recurse into nested classes.
        for child in node:
            if child.tag == "default":
                self._process(child, cls)

    def add_block(self, node):
        self._process(node, node.get("class", None))

    def get(self, cls, tag) -> dict:
        if cls is None:
            return {}
        return dict(self.table.get(cls, {}).get(tag, {}))


def _resolve_attrs(node, cls, defaults: _Defaults) -> dict:
    """Merge class defaults for this element tag with the element's own attrs."""
    attrs = defaults.get(cls, node.tag)
    attrs.update(node.attrib)
    return attrs


# ---------------------------------------------------------------------------
# Geometry / pose
# ---------------------------------------------------------------------------
def _pose(attrs: dict, angle_deg: bool) -> np.ndarray:
    pos = _floats(attrs.get("pos"), 3)
    if "quat" in attrs:
        return T.quat_wxyz_to_matrix(pos, _floats(attrs["quat"], 4))
    if "euler" in attrs:
        return T.euler_to_matrix(pos, _floats(attrs["euler"], 3), degrees=angle_deg)
    if "axisangle" in attrs:
        return T.axisangle_to_matrix(pos, _floats(attrs["axisangle"], 4), degrees=angle_deg)
    return T.translation(pos)


def _geom_color(attrs: dict, materials: dict) -> tuple:
    mat = attrs.get("material")
    if mat and mat in materials:
        return materials[mat][:3]
    rgba = _floats(attrs.get("rgba"), 4, 1.0) if "rgba" in attrs else _DEFAULT_GEOM_RGBA
    return tuple(rgba[:3])


def _build_geometry(attrs: dict, meshes: dict) -> Geometry | None:
    gtype = attrs.get("type", "sphere")  # MuJoCo's default geom type is sphere
    size = _floats(attrs.get("size"))
    if gtype == "mesh":
        name = attrs.get("mesh")
        info = meshes.get(name)
        if info is None:
            return None
        return Geometry(kind="mesh", mesh_path=info[0], scale=info[1])
    if gtype == "sphere":
        return Geometry(kind="sphere", radius=size[0] if size else 0.1)
    if gtype == "box":
        s = (size + [0.1, 0.1, 0.1])[:3]
        return Geometry(kind="box", box_size=(2 * s[0], 2 * s[1], 2 * s[2]))
    if gtype in ("cylinder", "capsule"):
        r = size[0] if size else 0.1
        half = size[1] if len(size) > 1 else r
        return Geometry(kind=gtype, radius=r, length=2 * half)
    if gtype == "ellipsoid":
        s = (size + [0.1, 0.1, 0.1])[:3]
        return Geometry(kind="ellipsoid", box_size=(2 * s[0], 2 * s[1], 2 * s[2]))
    return None  # plane / hfield / sdf — not rendered as link geometry


def _is_visual(attrs: dict, geom: Geometry) -> bool:
    """A geom is visual if it's a mesh, or an explicitly non-colliding primitive
    (``contype==0 and conaffinity==0`` — the MuJoCo-menagerie ``visual`` class)."""
    if geom.kind == "mesh":
        return True
    contype = int(float(attrs.get("contype", 1)))
    conaff = int(float(attrs.get("conaffinity", 1)))
    return contype == 0 and conaff == 0


# ---------------------------------------------------------------------------
# Body DFS
# ---------------------------------------------------------------------------
def _parse_body(body, parent_name, childclass, ctx, links, joints):
    defaults, meshes, materials, angle_deg = ctx
    name = body.get("name") or f"body_{len(links)}"
    cls = body.get("childclass", childclass)
    link = Link(name=name)

    for geom in body.findall("geom"):
        gcls = geom.get("class", cls)
        attrs = _resolve_attrs(geom, gcls, defaults)
        g = _build_geometry(attrs, meshes)
        if g is None:
            continue
        origin = _pose(attrs, angle_deg)
        if _is_visual(attrs, g):
            link.visuals.append(GeomInstance(origin, g, _geom_color(attrs, materials)))
        else:
            link.collisions.append(GeomInstance(origin, g, _COLLISION_COLOR))
    inertial = body.find("inertial")
    if inertial is not None and "mass" in inertial.attrib:
        link.mass = float(inertial.get("mass"))
    links[name] = link

    # Joint to the parent: body pos/quat is the joint origin; the body's own
    # <joint>/<freejoint> sets the DOF type. go2 has one joint per body.
    if parent_name is not None:
        origin = _pose(dict(body.attrib), angle_deg)
        jnodes = body.findall("joint")
        free = body.find("freejoint") is not None or any(
            j.get("type") == "free" for j in jnodes)
        if free:
            joints.append(Joint(name=f"{name}_free", type="floating", parent=parent_name,
                                child=name, origin=origin))
        elif jnodes:
            jn = jnodes[0]
            jattrs = _resolve_attrs(jn, jn.get("class", cls), defaults)
            jtype = jattrs.get("type", "hinge")
            kind = {"hinge": "revolute", "slide": "prismatic",
                    "ball": "fixed", "free": "floating"}.get(jtype, "fixed")
            axis = tuple(_floats(jattrs.get("axis", "0 0 1"), 3))
            rng = _floats(jattrs.get("range"), 2) if "range" in jattrs else [0.0, 0.0]
            if angle_deg and kind == "revolute":
                rng = [np.radians(rng[0]), np.radians(rng[1])]
            anchor = tuple(_floats(jattrs.get("pos"), 3))
            joints.append(Joint(name=jn.get("name") or f"{name}_joint", type=kind,
                                parent=parent_name, child=name, origin=origin,
                                axis=axis, lower=rng[0], upper=rng[1], anchor=anchor))
        else:
            joints.append(Joint(name=f"{name}_fixed", type="fixed", parent=parent_name,
                                child=name, origin=origin))

    for child in body.findall("body"):
        _parse_body(child, name, cls, ctx, links, joints)


def load_mjcf(path: str) -> RobotModel:
    """Parse a ``.xml`` (MJCF) file into a :class:`RobotModel`."""
    nodes = _collect_nodes(path)

    # --- compiler settings ---
    meshdir = ""
    angle_deg = False
    model_name = "robot"
    root_elem = ET.parse(os.path.abspath(path)).getroot()
    model_name = root_elem.get("model", "robot")
    for node, _ in nodes:
        if node.tag == "compiler":
            meshdir = node.get("meshdir", meshdir)
            angle_deg = node.get("angle", "radian") == "degree"

    # --- defaults ---
    defaults = _Defaults()
    for node, _ in nodes:
        if node.tag == "default":
            defaults.add_block(node)

    # --- assets (meshes keyed by name or file basename; materials by name) ---
    meshes: dict = {}
    materials: dict = {}
    for node, base_dir in nodes:
        if node.tag != "asset":
            continue
        for mesh in node.findall("mesh"):
            fname = mesh.get("file", "")
            mname = mesh.get("name") or os.path.splitext(os.path.basename(fname))[0]
            scale = tuple(_floats(mesh.get("scale"), 3, 1.0)) if mesh.get("scale") else (1.0, 1.0, 1.0)
            full = fname if os.path.isabs(fname) else os.path.join(
                base_dir, meshdir, fname)
            meshes[mname] = (os.path.normpath(full), scale)
        for mat in node.findall("material"):
            if "rgba" in mat.attrib:
                materials[mat.get("name", "")] = tuple(_floats(mat.get("rgba"), 4, 1.0))

    # --- worldbody DFS ---
    links: dict = {}
    joints: list = []
    ctx = (defaults, meshes, materials, angle_deg)
    root_name = None
    root_origin = T.identity()
    for node, _ in nodes:
        if node.tag != "worldbody":
            continue
        for body in node.findall("body"):
            if root_name is None:
                # First top-level body is the robot root; its pose is world-anchored.
                root_name = body.get("name") or "base"
                root_origin = _pose(dict(body.attrib), angle_deg)
                _parse_body(body, None, body.get("childclass", None), ctx, links, joints)
            else:
                # Additional top-level bodies (rare): attach to root as fixed.
                _parse_body(body, root_name, None, ctx, links, joints)

    if root_name is None:
        raise ValueError(f"No <body> found in worldbody of {path}")
    return RobotModel(name=model_name, links=links, joints=joints,
                      root=root_name, root_origin=root_origin)
