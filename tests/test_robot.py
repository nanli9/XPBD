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
H2_URDF = "/home/nan/Desktop/urdf/h2_description/H2.urdf"

_have_mjcf = os.path.exists(MJCF)
_have_urdf = os.path.exists(URDF)
_have_h2 = os.path.exists(H2_URDF)


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
    # foot_contacts off here to isolate the clustering: fixed joints are welded
    # into 13 clusters (base + 4×{hip,thigh,calf}), each revolute joint becoming
    # two coincident-anchor (hinge) constraints.
    phys = build_xpbd(m, base_static=True, device="cpu", substeps=10, iterations=6,
                      foot_contacts=False)
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
    """Shapes are taken directly from the file's collision primitives — and with
    foot_contacts (default) the *full* set is represented: go2's trunk/thighs are
    <box> → boxes, hips/calves are <cylinder> → cylinders (not capsules), and the
    foot <sphere> geoms appear as **sphere** proxies (the foot balls)."""
    from xpbd3d.robot.xpbd_build import build_xpbd

    m = load_mjcf(MJCF)
    phys = build_xpbd(m, base_static=False, device="cpu", substeps=10, iterations=6)
    kinds = set()
    for (_bidx, st, _he, _r, hl) in phys.proxies:
        kinds.add({0: "box", 2: "cylinder"}.get(st, "sphere" if hl < 1e-6 else "capsule"))
    assert "cylinder" in kinds           # hips/calves are real cylinders
    assert "box" in kinds                # trunk/thighs are boxes
    assert "sphere" in kinds             # the foot balls (<geom class="foot">)
    # one sphere proxy per foot → at least the 4 feet are present
    n_spheres = sum(1 for (_b, st, _h, _r, hl) in phys.proxies if st == 1 and hl < 1e-6)
    assert n_spheres >= 4
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
    # foot_contacts off to isolate the 13-cluster adjacency structure.
    phys = build_xpbd(m, base_static=True, device="cpu", substeps=10, iterations=6,
                      foot_contacts=False)
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
def test_actuated_robot_stands():
    """An actuated go2 (joint position servos + foot contacts) holds itself up
    under gravity with a *free* base, where the passive robot collapses to the
    floor. Runs on CPU (GPU-independent)."""
    from xpbd3d.robot.xpbd_build import build_xpbd

    m = load_mjcf(MJCF)
    q0 = {j.name: (0.9 if "thigh" in j.name.lower()
                   else -1.8 if "calf" in j.name.lower() else 0.0)
          for j in m.actuated_joints}

    def settled_base_height(actuation, feet):
        phys = build_xpbd(m, q0=q0, base_static=False, device="cpu", substeps=10,
                          iterations=8, friction=0.9, lin_damp=0.005, ang_damp=0.02,
                          actuation=actuation, foot_contacts=feet, start_clearance=0.04)
        bidx = phys.render[m.root][0]                  # base body, +y is up in sim frame
        for _ in range(90):
            phys.solver.step()
        pos = phys.solver.positions()
        return float(pos[bidx][1]), bool(np.isfinite(pos).all())

    h_passive, ok_p = settled_base_height(None, False)   # free hinges → collapses
    h_stand, ok_s = settled_base_height(1e-6, True)       # servos + feet → stands
    assert ok_p and ok_s
    assert h_stand > 0.18                                  # standing tall on its feet
    assert h_stand > h_passive + 0.1                       # clearly above the collapsed pose


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


@pytest.mark.skipif(not _have_h2, reason="H2 URDF not present")
def test_h2_collision_proxies_match_file():
    """The H2 humanoid's solver collision proxies mirror its URDF collision set
    **exactly**: one drawn/solved proxy per ``<collision>`` geom (the file's 2
    spheres + 2 cylinders stay exact primitives; mesh collisions become a tight
    box — the closest a box-only solver can come to an STL), and every link the
    file deliberately leaves *without* collision (wrists, hip-pitch, ankle-roll,
    waist, head-pitch — all commented out) becomes a **non-colliding** body that
    still carries mass and articulates but generates no contact at all. No
    bounding boxes are fabricated from visual meshes."""
    from xpbd3d.robot.xpbd_build import build_xpbd
    from xpbd3d.solver_6dof import NONCOLLIDING

    m = load_urdf(H2_URDF)
    file_kinds = [gi.geometry.kind for link in m.links.values() for gi in link.collisions]
    n_no_collision = sum(1 for link in m.links.values() if not link.collisions)

    phys = build_xpbd(m, base_static=True, device="cpu", substeps=10, iterations=6)
    s = phys.solver

    # exactly one proxy per file collision geom — nothing invented, nothing dropped.
    assert len(phys.proxies) == len(file_kinds)
    n_sphere = sum(1 for (_b, st, _h, _r, hl) in phys.proxies if st == 1 and hl < 1e-6)
    n_cyl = sum(1 for (_b, st, _h, _r, _hl) in phys.proxies if st == 2)
    assert n_sphere == file_kinds.count("sphere")        # 2 hip-roll spheres, exact
    assert n_cyl == file_kinds.count("cylinder")         # 2 knee cylinders, exact

    # links with no <collision> → non-colliding bodies (cat=0 + sentinel shape).
    cat = np.asarray(s._cat, np.uint64)
    stype = np.asarray(s._shape_type, np.int32)
    noncolliding = {i for i, c in enumerate(cat) if c == 0}
    assert len(noncolliding) == n_no_collision
    assert int((stype == NONCOLLIDING).sum()) == n_no_collision
    # none of those bodies appear among the drawn/solved proxies.
    assert all(b not in noncolliding for (b, *_rest) in phys.proxies)

    for _ in range(60):
        s.step()
    assert np.isfinite(s.positions()).all()


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
