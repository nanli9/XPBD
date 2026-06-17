"""URDF loader → :class:`RobotModel`.

A Python port of the parsing logic in the renderer's ``UrdfParser.cpp``:
materials → links (visual/collision: origin + geometry + material) → joints
(type, parent/child, origin, axis, limit), then build the kinematic tree (root =
the one link that never appears as a joint's child). Uses the stdlib
``xml.etree`` — no new XML dependency.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from . import transforms as T
from .model import Geometry, GeomInstance, Joint, Link, RobotModel

_TYPE_MAP = {
    "fixed": "fixed",
    "revolute": "revolute",
    "continuous": "continuous",
    "prismatic": "prismatic",
    "floating": "floating",
    "planar": "fixed",  # not articulated here; treat as rigid
}

_DEFAULT_COLOR = (0.7, 0.7, 0.75)


def _floats(text, n, default=0.0):
    if text is None:
        return [default] * n
    vals = [float(v) for v in text.replace(",", " ").split()]
    if len(vals) < n:
        vals += [default] * (n - len(vals))
    return vals[:n]


def _parse_origin(node) -> np.ndarray:
    if node is None:
        return T.identity()
    org = node.find("origin")
    if org is None:
        return T.identity()
    xyz = _floats(org.get("xyz"), 3)
    rpy = _floats(org.get("rpy"), 3)
    return T.rpy_to_matrix(xyz, rpy)


def _parse_materials(robot) -> dict:
    out = {}
    for mat in robot.findall("material"):
        name = mat.get("name", "")
        col = mat.find("color")
        if col is not None:
            rgba = _floats(col.get("rgba"), 4, 1.0)
            out[name] = tuple(rgba[:3])
    return out


def _parse_geometry(geom_node, urdf_file) -> Geometry | None:
    geo = geom_node.find("geometry")
    if geo is None:
        return None
    mesh = geo.find("mesh")
    if mesh is not None:
        path = T.resolve_urdf_mesh(urdf_file, mesh.get("filename", ""))
        scale = tuple(_floats(mesh.get("scale"), 3, 1.0)) if mesh.get("scale") else (1.0, 1.0, 1.0)
        return Geometry(kind="mesh", mesh_path=path, scale=scale)
    box = geo.find("box")
    if box is not None:
        return Geometry(kind="box", box_size=tuple(_floats(box.get("size"), 3, 1.0)))
    cyl = geo.find("cylinder")
    if cyl is not None:
        return Geometry(kind="cylinder", radius=float(cyl.get("radius", 0.1)),
                        length=float(cyl.get("length", 0.1)))
    sph = geo.find("sphere")
    if sph is not None:
        return Geometry(kind="sphere", radius=float(sph.get("radius", 0.1)))
    return None


def _color_of(node, materials) -> tuple:
    mat = node.find("material")
    if mat is None:
        return _DEFAULT_COLOR
    col = mat.find("color")
    if col is not None:
        return tuple(_floats(col.get("rgba"), 4, 1.0)[:3])
    name = mat.get("name", "")
    return materials.get(name, _DEFAULT_COLOR)


def load_urdf(path: str) -> RobotModel:
    """Parse a ``.urdf`` file into a :class:`RobotModel`."""
    tree = ET.parse(path)
    robot = tree.getroot()
    name = robot.get("name", "robot")
    materials = _parse_materials(robot)

    links: dict[str, Link] = {}
    for ln in robot.findall("link"):
        lname = ln.get("name", "")
        link = Link(name=lname)
        for vis in ln.findall("visual"):
            g = _parse_geometry(vis, path)
            if g is not None:
                link.visuals.append(GeomInstance(_parse_origin(vis), g, _color_of(vis, materials)))
        for col in ln.findall("collision"):
            g = _parse_geometry(col, path)
            if g is not None:
                link.collisions.append(GeomInstance(_parse_origin(col), g, (0.95, 0.55, 0.2)))
        inertial = ln.find("inertial")
        if inertial is not None and inertial.find("mass") is not None:
            link.mass = float(inertial.find("mass").get("value", 0.0))
        links[lname] = link

    joints: list[Joint] = []
    child_links = set()
    for jn in robot.findall("joint"):
        jtype = _TYPE_MAP.get(jn.get("type", "fixed"), "fixed")
        parent = jn.find("parent").get("link") if jn.find("parent") is not None else ""
        child = jn.find("child").get("link") if jn.find("child") is not None else ""
        axis_node = jn.find("axis")
        axis = tuple(_floats(axis_node.get("xyz"), 3)) if axis_node is not None else (1.0, 0.0, 0.0)
        lower = upper = 0.0
        lim = jn.find("limit")
        if lim is not None:
            lower = float(lim.get("lower", 0.0))
            upper = float(lim.get("upper", 0.0))
        joints.append(Joint(name=jn.get("name", ""), type=jtype, parent=parent,
                            child=child, origin=_parse_origin(jn), axis=axis,
                            lower=lower, upper=upper))
        child_links.add(child)

    roots = [n for n in links if n not in child_links]
    root = roots[0] if roots else next(iter(links))
    return RobotModel(name=name, links=links, joints=joints, root=root)
