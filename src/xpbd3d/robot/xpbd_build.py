"""Map a parsed :class:`RobotModel` onto the XPBD ``Solver6DOF`` — without
changing any solver code.

The solver only knows **boxes** (OBB contacts + floor) and a single compliant
**distance** ``add_joint`` (no spheres/cylinders, no hinge/motor). So this
builder approximates the robot within exactly those primitives:

* **Rigid clusters.** Links joined by *fixed* joints are welded into one rigid
  body (standard articulated-body practice), so cosmetic links (rotors, head,
  feet) ride with their parent. Each cluster → one box sized to the OBB-aligned
  AABB of its **collision** geometry (the cylinders/spheres a foot or hip uses
  collapse into that box — they have no solver primitive of their own).
* **Hinges from distance constraints.** A revolute joint becomes **two**
  coincident-anchor distance constraints placed along the hinge axis: pinning
  two points on the axis line leaves exactly one free DOF — rotation about that
  axis. One anchor pair alone would be a free ball joint; the second kills the
  two off-axis rotations.
* **Self-collision via a per-body category/mask bitmask.** Each cluster gets a
  unique category bit and a mask that clears its *joint-adjacent* neighbours'
  bits, so adjacent links (which overlap at the shared hinge anchor) never
  collide while every other link pair does — real inter-limb self-collision that
  doesn't fight the joints. A single ``group`` integer can't express this for a
  serial chain (hip must skip thigh, thigh must skip calf, yet hip *should* hit
  calf). ``self_collision=False`` falls back to the old all-off single-group
  filter. Every body collides with the floor regardless.

The solver is Y-up (gravity ``-Y``, floor on the Y plane) but robots are Z-up,
so the whole robot is rotated into a Y-up *sim frame* for the solve and rotated
back for rendering. Returns ``(solver, render)`` where ``render`` maps every link
to ``(body_index, offset)`` so the viewer can place all links from cluster poses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh.transformations as tf

from .model import FIXED, RobotModel
from . import transforms as T


# World(Z-up) → sim(Y-up): rotate -90° about X so +Z → +Y.
_TO_SIM = tf.rotation_matrix(-np.pi / 2.0, [1.0, 0.0, 0.0])
_TO_WORLD = np.linalg.inv(_TO_SIM)

def _quat_xyzw(M):
    w, x, y, z = tf.quaternion_from_matrix(M)
    return (float(x), float(y), float(z), float(w))


def _compose(quat_xyzw, pos):
    x, y, z, w = quat_xyzw
    M = tf.quaternion_matrix([w, x, y, z])
    M[:3, 3] = pos
    return M


@dataclass
class RobotPhysics:
    solver: object
    render: dict            # link_name -> (body_index, offset 4x4 in sim frame)
    proxies: list           # (body_index, shape_type, he, radius, half_len) per cluster
    base_static: bool


def _topo_links(model: RobotModel):
    order, stack = [], [model.root]
    while stack:
        n = stack.pop()
        order.append(n)
        for j in model._children.get(n, []):
            stack.append(j.child)
    return order


def _clusterize(model: RobotModel):
    """Weld links across fixed joints. Returns (cluster_of, roots, members,
    cross_joints) — cross_joints are the non-fixed tree edges between clusters."""
    cluster_of, members, roots, cross = {}, {}, [], []
    next_id = 0
    for link in _topo_links(model):
        j = model._by_child.get(link)
        if j is None:                          # model root
            cid = next_id; next_id += 1
            roots.append(link)
        elif j.type == FIXED:                  # weld into parent's cluster
            cid = cluster_of[j.parent]
        else:                                  # non-fixed → new cluster + edge
            cid = next_id; next_id += 1
            roots.append(link)
            cross.append(j)
        cluster_of[link] = cid
        members.setdefault(cid, []).append(link)
    return cluster_of, roots, members, cross


def _geom_volume(geo) -> float:
    if geo.kind == "box":
        return float(abs(np.prod(geo.box_size)))
    if geo.kind == "sphere":
        return float(geo.radius ** 3)
    if geo.kind in ("cylinder", "capsule"):
        return float(geo.radius ** 2 * geo.length)
    if geo.kind == "ellipsoid":
        return float(abs(np.prod(geo.box_size)))
    return 0.0


def _primary_collision_geom(model, member_links):
    """(owner_link, GeomInstance) of the largest-volume collision geom that maps
    to a solver primitive (box / sphere / cylinder / capsule), or ``None``."""
    best, best_vol = None, -1.0
    for lname in member_links:
        for gi in model.links[lname].collisions:
            if gi.geometry.kind not in ("box", "sphere", "cylinder", "capsule"):
                continue
            v = _geom_volume(gi.geometry)
            if v > best_vol:
                best_vol, best = v, (lname, gi)
    return best


def _cluster_aabb(model, member_links, root_name, fk_sim):
    """AABB (lo, hi) over the cluster's collision geometry in the root sim frame
    (visual as a fallback)."""
    Minv_root = np.linalg.inv(fk_sim[root_name])
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    found = False
    for use_collision in (True, False):
        for lname in member_links:
            geoms = model.links[lname].collisions if use_collision else model.links[lname].visuals
            for gi in geoms:
                m = gi.geometry.to_trimesh()
                if m is None or len(m.vertices) == 0:
                    continue
                W = Minv_root @ fk_sim[lname] @ gi.origin
                v = np.asarray(m.vertices, np.float64) @ W[:3, :3].T + W[:3, 3]
                lo = np.minimum(lo, v.min(axis=0))
                hi = np.maximum(hi, v.max(axis=0))
                found = True
        if found:
            break
    if not found:
        return np.full(3, -0.02), np.full(3, 0.02)
    return lo, hi


def _cluster_shape(model, member_links, root_name, fk_sim):
    """Map a rigid cluster onto a solver shape, taken **directly from the file's
    collision geometry** — the largest collision primitive's own dimensions and
    pose, not a fitted bounding volume. A ``<cylinder>``/``<capsule>`` geom →
    capsule (its radius/length, axis = the geom's local +Z), ``<sphere>`` → sphere,
    ``<box>`` → box. Falls back to an AABB box only for mesh-only clusters.
    Returns ``{kind, B0 (4x4 body frame, sim), half | radius | half_len}``.
    """
    prim = _primary_collision_geom(model, member_links)
    if prim is not None:
        owner, gi = prim
        B0 = fk_sim[owner] @ gi.origin          # the geom's own pose in the sim frame
        g = gi.geometry
        if g.kind == "box":
            return {"kind": "box", "B0": B0, "half": 0.5 * np.asarray(g.box_size, float)}
        if g.kind == "sphere":
            return {"kind": "sphere", "B0": B0, "radius": float(g.radius)}
        if g.kind == "cylinder":            # exact cylinder (flat caps), axis = geom +Z
            return {"kind": "cylinder", "B0": B0, "radius": float(g.radius),
                    "half_len": 0.5 * float(g.length)}
        # capsule → solver capsule along the geom's local +Z axis
        return {"kind": "capsule", "B0": B0, "radius": float(g.radius),
                "half_len": 0.5 * float(g.length)}
    # mesh-only cluster: no usable primitive → AABB box fallback
    lo, hi = _cluster_aabb(model, member_links, root_name, fk_sim)
    center = 0.5 * (lo + hi)
    half = np.maximum(0.5 * (hi - lo), 0.005)
    return {"kind": "box", "B0": fk_sim[root_name] @ T.translation(center), "half": half}


def build_xpbd(model: RobotModel, q0: dict | None = None, base_static: bool = True,
               density: float = 700.0, hinge_delta: float = 0.06,
               start_clearance: float = 0.06, group: int = 7,
               self_collision: bool = True,
               actuation: float | None = None, drive_lever: float = 0.06,
               foot_contacts: bool = False,
               **solver_kwargs) -> RobotPhysics:
    """Map ``model`` onto ``Solver6DOF``. See module docstring for the rigid-
    cluster / hinge / self-collision design.

    ``actuation`` turns each passive hinge into a **position servo** that holds
    its build-pose angle (``q0``): a compliant off-axis anchor pair acts as a
    torsional spring about the hinge axis (lever ``drive_lever``), built from the
    existing ``add_joint`` — no solver change. ``None`` = passive free hinges (the
    legs relax under gravity); ``0.0`` = rigid lock (frozen pose); a small value
    (~1e-4) = a stiff-but-springy motor that lets the robot **stand stably** while
    still flexing under load. The target pose is whatever ``q0`` encodes."""
    from ..solver_6dof import Solver6DOF, FULL64

    # FK in world (Z-up), lift so the lowest collision vertex starts at clearance.
    fk_world = model.forward_kinematics(q0)
    min_up = np.inf
    for lname, link in model.links.items():
        for gi in (link.collisions or link.visuals):
            m = gi.geometry.to_trimesh()
            if m is None or len(m.vertices) == 0:
                continue
            W = fk_world[lname] @ gi.origin
            z = (np.asarray(m.vertices, np.float64) @ W[:3, :3].T + W[:3, 3])[:, 2]
            min_up = min(min_up, float(z.min()))
    dz = (start_clearance - min_up) if np.isfinite(min_up) else 0.0
    lift = T.translation((0.0, 0.0, dz))
    fk_sim = {k: _TO_SIM @ lift @ M for k, M in fk_world.items()}

    cluster_of, roots, members, cross = _clusterize(model)

    solver_kwargs.setdefault("gravity", (0.0, -9.81, 0.0))
    solver_kwargs.setdefault("floor_y", 0.0)
    solver = Solver6DOF(**solver_kwargs)

    # Self-collision via per-body category/mask bits needs a unique bit per
    # cluster, so it only applies when there are <= 64 clusters. With it on, all
    # links share group 0 (group filter inert) and joint-adjacent pairs are
    # excluded by the bitmask; off, the old single-group filter disables all
    # link-vs-link contact.
    n_clusters = len(members)
    use_bits = self_collision and n_clusters <= 64
    body_group = 0 if use_bits else group

    # One shape (box / capsule / sphere) per rigid cluster.
    cluster_body, cluster_frame, proxies = {}, {}, []
    render = {}
    for cid in sorted(members):
        root = roots[cid]
        shp = _cluster_shape(model, members[cid], root, fk_sim)
        B0 = shp["B0"]                          # body frame = the collision geom's pose
        cw = B0[:3, 3]
        R_body = B0[:3, :3]
        mass = sum(model.links[l].mass for l in members[cid])
        is_static = base_static and root == model.root
        color = (0.30, 0.65, 0.90)
        if shp["kind"] == "cylinder":
            r, hl = shp["radius"], shp["half_len"]
            if mass <= 0.0:
                mass = max(0.02, density * float(np.pi * r * r * 2.0 * hl))
            rb = solver.add_cylinder(tuple(cw), r, hl, mass=mass,
                                     quaternion=_quat_xyzw(B0), static=is_static,
                                     color=color, group=body_group)
            proxies.append((None, 2, None, r, hl))
        elif shp["kind"] == "capsule":
            r, hl = shp["radius"], shp["half_len"]
            if mass <= 0.0:
                vol = np.pi * r * r * (2.0 * hl) + 4.0 / 3.0 * np.pi * r ** 3
                mass = max(0.02, density * float(vol))
            rb = solver.add_capsule(tuple(cw), r, hl, mass=mass,
                                    quaternion=_quat_xyzw(B0), static=is_static,
                                    color=color, group=body_group)
            proxies.append((None, 1, None, r, hl))
        elif shp["kind"] == "sphere":
            r = shp["radius"]
            if mass <= 0.0:
                mass = max(0.02, density * float(4.0 / 3.0 * np.pi * r ** 3))
            rb = solver.add_sphere(tuple(cw), r, mass=mass, static=is_static,
                                   color=color, group=body_group)
            proxies.append((None, 1, None, r, 0.0))
        else:
            half = shp["half"]
            if mass <= 0.0:
                mass = max(0.02, density * float(8.0 * half[0] * half[1] * half[2]))
            rb = solver.add_box(tuple(cw), tuple(half), mass=mass,
                                quaternion=_quat_xyzw(B0), static=is_static,
                                color=color)
            solver._group[rb.index] = int(body_group)      # box: set the group manually
            rb.group = int(body_group)
            proxies.append((None, 0, tuple(float(x) for x in half), 0.0, 0.0))
        proxies[-1] = (rb.index,) + proxies[-1][1:]
        cluster_body[cid] = rb.index
        cluster_frame[cid] = (np.asarray(cw), R_body)
        B0inv = np.linalg.inv(B0)
        for lname in members[cid]:
            render[lname] = (rb.index, B0inv @ fk_sim[lname])

    # Hinge (or ball) constraints across cluster boundaries.
    adj = {bidx: set() for bidx in cluster_body.values()}
    for j in cross:
        ca, cb = cluster_of[j.parent], cluster_of[j.child]
        ia, ib = cluster_body[ca], cluster_body[cb]
        adj[ia].add(ib); adj[ib].add(ia)        # joint-adjacent → exclude from self-collision
        cw_a, R_a = cluster_frame[ca]
        cw_b, R_b = cluster_frame[cb]
        Mchild = fk_sim[j.child]                           # child link == cluster B root
        anchor = np.asarray(j.anchor, np.float64)
        pts = [Mchild @ np.append(anchor, 1.0)]
        ax_world = None
        if j.actuated and j.type != "prismatic":
            ax = np.asarray(j.axis, np.float64)
            n = np.linalg.norm(ax)
            if n > 1e-9:
                ax = ax / n
                ax_world = Mchild[:3, :3] @ ax             # hinge axis in the sim frame
                pts.append(Mchild @ np.append(anchor + hinge_delta * ax, 1.0))
        for Pw in pts:
            p = Pw[:3]
            anchor_a = R_a.T @ (p - cw_a)
            anchor_b = R_b.T @ (p - cw_b)
            solver.add_joint(ia, ib, compliance=0.0, rest_length=0.0,
                             anchor_a=tuple(anchor_a), anchor_b=tuple(anchor_b))

        # Actuation: a compliant off-axis anchor pair → a torsional spring that
        # holds the joint at its q0 angle. Coincident at the build pose, it pulls
        # the limb back when it twists about the hinge axis (a position servo).
        if actuation is not None and ax_world is not None:
            e = np.array([1.0, 0.0, 0.0]) if abs(ax_world[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            t = np.cross(ax_world, e)
            t = t / (np.linalg.norm(t) + 1e-12)
            D = pts[0][:3] + drive_lever * t              # off-axis drive point
            aa = R_a.T @ (D - cw_a)
            ab = R_b.T @ (D - cw_b)
            solver.add_joint(ia, ib, compliance=float(actuation), rest_length=0.0,
                             anchor_a=tuple(aa), anchor_b=tuple(ab))

    # Secondary contact shapes: a cluster's non-primary primitive collision geoms
    # (e.g. the foot sphere when the calf cylinder is the primary) become small
    # bodies rigidly welded to the cluster's main body, so the robot stands on its
    # **feet** — not just on whichever single proxy happened to be largest.
    extra_excl = {bidx: set() for bidx in cluster_body.values()}
    sec_of_main = {}                                    # secondary body -> its main body
    if foot_contacts:
        for cid in sorted(members):
            main_idx = cluster_body[cid]
            cw_main, R_main = cluster_frame[cid]
            primary = _primary_collision_geom(model, members[cid])
            for lname in members[cid]:
                for gi in model.links[lname].collisions:
                    g = gi.geometry
                    if g.kind not in ("box", "sphere", "cylinder", "capsule"):
                        continue
                    if primary is not None and lname == primary[0] and gi is primary[1]:
                        continue                       # already the main body
                    W = fk_sim[lname] @ gi.origin
                    c = W[:3, 3]
                    Rw = W[:3, :3]
                    mass = max(0.05, 0.3 * density * _geom_volume(g))  # light contact proxy
                    if g.kind == "box":
                        half = 0.5 * np.asarray(g.box_size, float)
                        sb = solver.add_box(tuple(c), tuple(half), mass=mass,
                                            quaternion=_quat_xyzw(W), color=(0.95, 0.6, 0.2))
                        solver._group[sb.index] = body_group; sb.group = body_group
                        proxies.append((sb.index, 0, tuple(float(x) for x in half), 0.0, 0.0))
                        Rsb = Rw
                    elif g.kind == "sphere":
                        sb = solver.add_sphere(tuple(c), float(g.radius), mass=mass,
                                               color=(0.95, 0.6, 0.2), group=body_group)
                        proxies.append((sb.index, 1, None, float(g.radius), 0.0))
                        Rsb = np.eye(3)                # sphere body frame is world-aligned
                    elif g.kind == "cylinder":
                        hl = 0.5 * float(g.length)
                        sb = solver.add_cylinder(tuple(c), float(g.radius), hl, mass=mass,
                                                 quaternion=_quat_xyzw(W), color=(0.95, 0.6, 0.2),
                                                 group=body_group)
                        proxies.append((sb.index, 2, None, float(g.radius), hl))
                        Rsb = Rw
                    else:
                        hl = 0.5 * float(g.length)
                        sb = solver.add_capsule(tuple(c), float(g.radius), hl, mass=mass,
                                                quaternion=_quat_xyzw(W), color=(0.95, 0.6, 0.2),
                                                group=body_group)
                        proxies.append((sb.index, 1, None, float(g.radius), hl))
                        Rsb = Rw
                    # rigid weld to the main body: 3 non-collinear coincident points.
                    for off in (np.zeros(3), 0.03 * Rw[:, 0], 0.03 * Rw[:, 1]):
                        P = c + off
                        solver.add_joint(main_idx, sb.index, compliance=0.0, rest_length=0.0,
                                         anchor_a=tuple(R_main.T @ (P - cw_main)),
                                         anchor_b=tuple(Rsb.T @ (P - c)))
                    extra_excl[main_idx].add(sb.index)
                    sec_of_main[sb.index] = main_idx

    # Self-collision filter: each cluster gets a unique category bit; its mask
    # clears its own bit and every joint-adjacent neighbour's bit, so adjacent
    # links (which overlap at the shared hinge anchor) never collide while all
    # other link pairs do — real self-collision without fighting the joints.
    # Secondary contact bodies inherit their main body's exclusions (so a foot
    # never collides its own calf/thigh) but otherwise collide the world.
    use_bits = use_bits and solver.num_bodies <= 64
    if use_bits:
        for bidx, nbrs in adj.items():
            excl = (1 << bidx)
            for nb in nbrs:
                excl |= (1 << nb)
            for sb in extra_excl[bidx]:
                excl |= (1 << sb)
            solver.set_collision_filter(bidx, 1 << bidx, FULL64 & ~excl)
        for sb, main_idx in sec_of_main.items():
            excl = (1 << sb) | (1 << main_idx)
            for nb in adj[main_idx]:
                excl |= (1 << nb)
            for sib in extra_excl[main_idx]:
                excl |= (1 << sib)
            solver.set_collision_filter(sb, 1 << sb, FULL64 & ~excl)

    return RobotPhysics(solver=solver, render=render, proxies=proxies,
                        base_static=base_static)


def read_state(phys: RobotPhysics):
    """One host readback of the solver's body poses, to share across the
    per-frame render helpers (keeps the rendering sync to a single transfer)."""
    return phys.solver.positions(), phys.solver.orientations()


def link_world_transforms(phys: RobotPhysics, state=None) -> dict:
    """Current world (Z-up) 4x4 for every link, from the solver's cluster poses.
    Pass ``state=read_state(phys)`` to reuse one readback for multiple helpers."""
    pos, quat = read_state(phys) if state is None else state
    out = {}
    for lname, (bidx, offset) in phys.render.items():
        B_sim = _compose(tuple(quat[bidx]), pos[bidx])
        out[lname] = _TO_WORLD @ B_sim @ offset
    return out


def proxy_world_poses(phys: RobotPhysics, state=None):
    """(center_world, wxyz_world) per collision proxy, aligned with ``phys.proxies``
    — for drawing the exact shapes (box/capsule/sphere) the solver simulates."""
    pos, quat = read_state(phys) if state is None else state
    out = []
    for (bidx, _st, _he, _r, _hl) in phys.proxies:
        B_world = _TO_WORLD @ _compose(tuple(quat[bidx]), pos[bidx])
        w, x, y, z = tf.quaternion_from_matrix(B_world)
        out.append((tuple(B_world[:3, 3]), (w, x, y, z)))
    return out
