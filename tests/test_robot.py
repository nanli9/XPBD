"""Robot loader tests — URDF + MJCF parse into a common model with valid FK.

Visualisation milestone only (no solver). Skips gracefully if the sibling robot
description repos aren't present.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from xpbd3d.robot import load_mjcf, load_urdf

MJCF = "/home/nan/Desktop/unitree_robots/go2/go2.xml"
URDF = "/home/nan/Desktop/urdf/go2_description/urdf/go2_description.urdf"

_have_mjcf = os.path.exists(MJCF)
_have_urdf = os.path.exists(URDF)


@pytest.mark.skipif(not _have_mjcf, reason="go2 MJCF not present")
def test_mjcf_structure():
    m = load_mjcf(MJCF)
    assert m.root == "base_link"
    # 12 actuated leg joints (3 per leg × 4 legs), all hinge/revolute.
    assert len(m.actuated_joints) == 12
    assert all(j.type == "revolute" for j in m.actuated_joints)
    # base body is anchored at the MJCF spawn height.
    assert m.root_origin[2, 3] == pytest.approx(0.445, abs=1e-6)
    # base_link carries both the visual meshes and primitive collision shapes.
    base = m.links["base_link"]
    assert len(base.visuals) == 5
    assert len(base.collisions) == 3


@pytest.mark.skipif(not _have_urdf, reason="go2 URDF not present")
def test_urdf_structure():
    u = load_urdf(URDF)
    assert u.root == "base"
    assert len(u.actuated_joints) == 12
    assert all(j.type == "revolute" for j in u.actuated_joints)


@pytest.mark.skipif(not _have_mjcf, reason="go2 MJCF not present")
def test_fk_finite_and_moves():
    m = load_mjcf(MJCF)
    fk0 = m.forward_kinematics()
    assert all(np.isfinite(M).all() and M.shape == (4, 4) for M in fk0.values())
    # Bending a knee rotates its calf link about the knee anchor: the calf frame
    # origin stays put but its orientation changes (and child links translate).
    knee = next(j for j in m.actuated_joints if "calf" in j.name)
    fk1 = m.forward_kinematics({knee.name: 1.0})
    assert not np.allclose(fk0[knee.child][:3, :3], fk1[knee.child][:3, :3])
    # The foot, a child of the calf, must actually translate.
    foot = next((j.child for j in m.joints if j.parent == knee.child), None)
    if foot is not None:
        assert not np.allclose(fk0[foot][:3, 3], fk1[foot][:3, 3])


@pytest.mark.skipif(not _have_mjcf, reason="go2 MJCF not present")
def test_xpbd_build_and_step_stable():
    """Map the robot onto Solver6DOF and confirm it steps stably (no explosion).
    Runs on CPU so it's GPU-independent."""
    from xpbd3d.robot.xpbd_build import build_xpbd, link_world_transforms

    m = load_mjcf(MJCF)
    phys = build_xpbd(m, base_static=True, device="cpu", substeps=10, iterations=6)
    # Fixed joints are welded: 13 clusters (base + 4×{hip,thigh,calf}), and each
    # revolute joint becomes two coincident-anchor (hinge) constraints.
    assert phys.solver.num_bodies == 13
    assert phys.solver.num_joints == 2 * len(m.actuated_joints)
    for _ in range(60):
        phys.solver.step()
    pos = phys.solver.positions()
    assert np.isfinite(pos).all()
    assert np.abs(pos).max() < 10.0  # bounded — nothing flew off
    world = link_world_transforms(phys)
    assert all(np.isfinite(M).all() for M in world.values())


@pytest.mark.skipif(not _have_mjcf, reason="go2 MJCF not present")
def test_xpbd_build_uses_file_shapes():
    """Shapes are taken directly from the file's collision primitives: go2's
    trunk/thighs are <box> → boxes, its hips/calves are <cylinder> → cylinders
    (not capsule approximations)."""
    from xpbd3d.robot.xpbd_build import build_xpbd

    m = load_mjcf(MJCF)
    phys = build_xpbd(m, base_static=False, device="cpu", substeps=10, iterations=6)
    kinds = set()
    for (_bidx, st, _he, _r, hl) in phys.proxies:
        kinds.add({0: "box", 2: "cylinder"}.get(st, "sphere" if hl < 1e-6 else "capsule"))
    assert "cylinder" in kinds           # hips/calves are real cylinders
    assert "box" in kinds                # trunk/thighs are boxes
    for _ in range(40):
        phys.solver.step()
    assert np.isfinite(phys.solver.positions()).all()


@pytest.mark.skipif(not _have_mjcf, reason="go2 MJCF not present")
def test_self_collision_excludes_only_adjacent():
    """With self-collision on, every link shares group 0 and the per-body
    category/mask bitmask excludes exactly the joint-adjacent cluster pairs (one
    per revolute joint) while leaving all other link pairs collidable. Turning it
    off reverts to the old single-group (all self-collision disabled)."""
    from xpbd3d.robot.xpbd_build import build_xpbd

    m = load_mjcf(MJCF)
    phys = build_xpbd(m, base_static=True, device="cpu", substeps=10, iterations=6)
    s = phys.solver
    assert set(s._group) == {0}                  # group filter inert; bitmask governs
    cat = np.asarray(s._cat, np.uint64)
    mask = np.asarray(s._mask, np.uint64)
    n = s.num_bodies
    live = excluded = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (cat[i] & mask[j]) == np.uint64(0) or (cat[j] & mask[i]) == np.uint64(0):
                excluded += 1
            else:
                live += 1
    # one excluded pair per cross (revolute) joint; everything else collides.
    assert excluded == len(m.actuated_joints)    # 12 adjacent pairs
    assert live == n * (n - 1) // 2 - excluded   # the other 66 pairs are live
    assert live > 0

    off = build_xpbd(m, base_static=True, device="cpu", self_collision=False,
                     substeps=10, iterations=6)
    assert set(off.solver._group) == {7}         # falls back to single-group filter


@pytest.mark.skipif(not _have_mjcf, reason="go2 MJCF not present")
def test_registry_discovers_models():
    """Model discovery finds loadable robots with unique labels, and go2 (the
    model both viewers default to) is among them."""
    from xpbd3d.robot.registry import discover_models, entry_for_path

    models = discover_models()
    assert len(models) >= 1
    labels = [e.label for e in models]
    assert len(labels) == len(set(labels))            # labels are unique
    assert all(os.path.exists(e.path) and e.links >= 1 for e in models)
    assert any(e.label == "go2 [mjcf]" for e in models)
    # the CLI default path resolves back to a discovered entry
    assert entry_for_path(models, "mjcf", MJCF) is not None


@pytest.mark.skipif(not (_have_mjcf and _have_urdf), reason="need both descriptions")
def test_mjcf_urdf_leg_kinematics_agree():
    """Same robot, two formats: relative leg geometry must match (URDF has no
    base offset, so compare positions relative to each model's base)."""
    m = load_mjcf(MJCF)
    u = load_urdf(URDF)
    fm = m.forward_kinematics()
    fu = u.forward_kinematics()
    bm = fm["base_link"][:3, 3]
    bu = fu["base"][:3, 3]
    for ln in ("FL_hip", "FL_thigh", "FL_calf", "RR_hip", "RR_thigh", "RR_calf"):
        rel_m = fm[ln][:3, 3] - bm
        rel_u = fu[ln][:3, 3] - bu
        assert np.allclose(rel_m, rel_u, atol=2e-3), (ln, rel_m, rel_u)
