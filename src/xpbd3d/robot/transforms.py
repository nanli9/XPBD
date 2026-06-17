"""Transform and asset-path helpers for the robot loaders.

Rotation math is delegated to ``trimesh.transformations`` (already a project
dependency) so we don't re-derive Euler/quaternion conventions by hand:

* **URDF** uses ``xyz`` + ``rpy`` — extrinsic roll-pitch-yaw, i.e.
  ``R = Rz(yaw) · Ry(pitch) · Rx(roll)`` → ``euler_matrix(r, p, y, "sxyz")``.
* **MJCF** uses ``pos`` + ``quat`` (**wxyz**), with optional ``euler`` /
  ``axisangle``; ``compiler angle="radian|degree"`` controls units.

All functions return 4x4 ``float64`` homogeneous matrices.
"""

from __future__ import annotations

import os

import numpy as np
import trimesh.transformations as tf


def identity() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def translation(xyz) -> np.ndarray:
    return tf.translation_matrix(np.asarray(xyz, dtype=np.float64))


def rpy_to_matrix(xyz, rpy) -> np.ndarray:
    """URDF ``origin``: translation ``xyz`` then extrinsic RPY rotation."""
    T = tf.euler_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]), axes="sxyz")
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


def quat_wxyz_to_matrix(pos, wxyz) -> np.ndarray:
    """MJCF ``pos`` + ``quat`` (w, x, y, z)."""
    T = tf.quaternion_matrix(np.asarray(wxyz, dtype=np.float64))
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def euler_to_matrix(pos, euler, seq: str = "xyz", degrees: bool = False) -> np.ndarray:
    """MJCF ``euler``. MuJoCo's ``eulerseq`` defaults to intrinsic ``xyz``.

    A lowercase axis letter means rotation about the moving (intrinsic) frame,
    uppercase about the fixed (extrinsic) frame — same convention as
    ``trimesh.transformations`` ``"r..."`` / ``"s..."`` prefixes.
    """
    e = np.asarray(euler, dtype=np.float64)
    if degrees:
        e = np.radians(e)
    prefix = "r" if seq[0].islower() else "s"
    axes = prefix + seq.lower()
    T = tf.euler_matrix(e[0], e[1], e[2], axes=axes)
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def axisangle_to_matrix(pos, axisangle, degrees: bool = False) -> np.ndarray:
    """MJCF ``axisangle`` = (ax, ay, az, angle)."""
    a = np.asarray(axisangle, dtype=np.float64)
    angle = np.radians(a[3]) if degrees else a[3]
    axis = a[:3]
    if np.linalg.norm(axis) < 1e-12:
        T = np.eye(4)
    else:
        T = tf.rotation_matrix(float(angle), axis)
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def axis_angle_motion(axis, angle: float) -> np.ndarray:
    """4x4 rotation of ``angle`` about ``axis`` through the origin (joint motion)."""
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return identity()
    return tf.rotation_matrix(float(angle), axis / n)


def prismatic_motion(axis, distance: float) -> np.ndarray:
    """4x4 translation of ``distance`` along ``axis`` (prismatic joint motion)."""
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return identity()
    return tf.translation_matrix(axis / n * float(distance))


# ---------------------------------------------------------------------------
# Asset-path resolution
# ---------------------------------------------------------------------------
def resolve_urdf_mesh(urdf_file: str, uri: str) -> str:
    """Resolve a URDF mesh ``filename``.

    Mirrors ``ResolveUrdfAssetPath`` in the renderer's ``UrdfParser.cpp``:
    ``package://<pkg>/<rest>`` is resolved by walking up from the URDF file to a
    directory named ``<pkg>`` (or one containing it / a ``package.xml``); plain
    paths are taken relative to the URDF file's directory.
    """
    base_dir = os.path.dirname(os.path.abspath(urdf_file))
    if uri.startswith("package://"):
        rest = uri[len("package://"):]
        slash = rest.find("/")
        pkg = rest[:slash] if slash >= 0 else rest
        sub = rest[slash + 1:] if slash >= 0 else ""
        cur = base_dir
        while cur and cur != os.path.dirname(cur):
            if os.path.basename(cur) == pkg:
                return os.path.normpath(os.path.join(cur, sub))
            cand = os.path.join(cur, pkg)
            if os.path.isdir(cand):
                return os.path.normpath(os.path.join(cand, sub))
            cur = os.path.dirname(cur)
        # Fall back: assume the package root is the URDF's parent's parent.
        return os.path.normpath(os.path.join(os.path.dirname(base_dir), sub))
    if uri.startswith("file://"):
        uri = uri[len("file://"):]
    if os.path.isabs(uri):
        return os.path.normpath(uri)
    return os.path.normpath(os.path.join(base_dir, uri))


def resolve_mjcf_mesh(xml_file: str, meshdir: str, filename: str) -> str:
    """Resolve an MJCF mesh ``file`` against ``compiler meshdir`` (relative to
    the model XML's directory)."""
    if os.path.isabs(filename):
        return os.path.normpath(filename)
    base_dir = os.path.dirname(os.path.abspath(xml_file))
    root = base_dir if os.path.isabs(meshdir) else os.path.join(base_dir, meshdir)
    return os.path.normpath(os.path.join(root, filename))
