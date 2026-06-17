"""Discover the robot descriptions available on disk so a viewer can offer a
model dropdown instead of a hard-coded ``--robot``.

Both Unitree description repos sit beside this one:

* MJCF  — ``/home/nan/Desktop/unitree_robots/<robot>/<robot>.xml`` (+ ``g1`` has
  ``g1_23dof.xml`` / ``g1_29dof.xml``, ``h1_2`` has ``h1_2_handless.xml``; the
  ``scene*.xml`` / ``*_terrain.xml`` files are world wrappers, not robots).
* URDF  — ``/home/nan/Desktop/urdf/<name>_description/urdf/*.urdf`` (file names
  are irregular: ``go2_description.urdf`` but ``a1.urdf``, ``h1.urdf`` + a
  ``h1_with_hand.urdf`` variant; some dirs ship no flat ``.urdf`` at all).

So discovery globs the actual files (no name assumptions) and, by default,
*validates* each by parsing it into a :class:`RobotModel` — cheap, since the
loaders only read XML and resolve mesh paths (meshes load lazily at render). Only
models that parse with at least one link are offered.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

MJCF_ROOT = "/home/nan/Desktop/unitree_robots"
URDF_ROOT = "/home/nan/Desktop/urdf"

# MJCF files that wrap a robot in a world (floor/lights/terrain) rather than
# defining the robot itself — skip them as standalone entries.
_MJCF_SKIP = ("scene", "terrain", "map", "qrc")


@dataclass(frozen=True)
class ModelEntry:
    label: str          # unique, human-readable (e.g. "go2 [mjcf]")
    fmt: str            # "mjcf" | "urdf"
    path: str           # absolute path to the .xml / .urdf
    links: int = 0
    actuated: int = 0


def _mjcf_files():
    if not os.path.isdir(MJCF_ROOT):
        return []
    out = []
    for folder in sorted(os.listdir(MJCF_ROOT)):
        d = os.path.join(MJCF_ROOT, folder)
        if not os.path.isdir(d):
            continue
        for xml in sorted(glob.glob(os.path.join(d, "*.xml"))):
            stem = os.path.splitext(os.path.basename(xml))[0]
            if any(k in stem.lower() for k in _MJCF_SKIP):
                continue
            out.append((stem, xml))
    return out


def _urdf_files():
    if not os.path.isdir(URDF_ROOT):
        return []
    out = []
    for folder in sorted(os.listdir(URDF_ROOT)):
        d = os.path.join(URDF_ROOT, folder)
        if not os.path.isdir(d):
            continue
        urdfs = (sorted(glob.glob(os.path.join(d, "urdf", "*.urdf")))
                 or sorted(glob.glob(os.path.join(d, "*.urdf"))))
        for u in urdfs:
            stem = os.path.splitext(os.path.basename(u))[0]
            # tidy the common "<name>_description" stem down to "<name>"
            label = stem[:-12] if stem.endswith("_description") else stem
            out.append((label or folder, u))
    return out


def discover_models(validate: bool = True) -> list[ModelEntry]:
    """All loadable robot models across both description repos, sorted by label.

    With ``validate`` (default) each candidate is parsed and kept only if it
    yields >= 1 link; the link/actuated counts are filled in. Labels are made
    unique by suffixing the format and, if still colliding, an index."""
    from . import load_mjcf, load_urdf

    raw = [("mjcf", s, p) for s, p in _mjcf_files()] + \
          [("urdf", s, p) for s, p in _urdf_files()]

    entries: list[ModelEntry] = []
    seen: dict[str, int] = {}
    for fmt, stem, path in raw:
        links = actuated = 0
        if validate:
            try:
                m = (load_mjcf if fmt == "mjcf" else load_urdf)(path)
            except Exception:
                continue
            if len(m.links) < 1:
                continue
            links, actuated = len(m.links), len(m.actuated_joints)
        label = f"{stem} [{fmt}]"
        if label in seen:
            seen[label] += 1
            label = f"{stem} [{fmt}] ({seen[label]})"
        else:
            seen[label] = 0
        entries.append(ModelEntry(label, fmt, path, links, actuated))
    entries.sort(key=lambda e: e.label.lower())
    return entries


def entry_for_path(entries, fmt: str, path: str) -> ModelEntry | None:
    """The discovered entry matching an explicit ``path`` (so a CLI-launched
    model is preselected in the dropdown), or ``None``."""
    ap = os.path.abspath(path)
    for e in entries:
        if e.fmt == fmt and os.path.abspath(e.path) == ap:
            return e
    return None
