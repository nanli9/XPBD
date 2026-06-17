"""Headless diagnostic for the `hang` robot scene.

Builds the *exact* physics the viser demo builds (`sim_robot_xpbd.py --scene
hang`), steps it without a browser, and instruments the three things the user
is asking about:

1. **Joint residuals** — for every hinge (a pair of coincident-anchor distance
   constraints) measure ||pa - pb||. ~0 means the hinge holds; growth means the
   articulation is separating / flying apart.
2. **Energy & speeds** — max linear speed, max |omega|, and total kinetic energy
   per frame. A pinned-base robot should *lose* energy and settle; growth is the
   "swings and flies around" the user reports.
3. **Self-collision** — how many body-body contact pairs survive the broad-phase
   group filter. If 0, links pass through each other freely (the persistent
   self-intersection).

Run:  uv run python examples/diag_robot_hang.py
"""

from __future__ import annotations

import argparse

import numpy as np

from xpbd3d.robot import load_mjcf
from xpbd3d.robot.xpbd_build import build_xpbd, read_state

MJCF = "/home/nan/Desktop/unitree_robots/go2/go2.xml"


def home_pose(model) -> dict:
    q = {}
    for j in model.actuated_joints:
        n = j.name.lower()
        v = 0.9 if "thigh" in n else (-1.8 if ("calf" in n or "knee" in n) else 0.0)
        if j.lower < j.upper:
            v = float(np.clip(v, j.lower, j.upper))
        q[j.name] = v
    return q


def quat_rotate(q_xyzw, v):
    x, y, z, w = q_xyzw
    # rotate v by quaternion (xyzw)
    uv = np.cross([x, y, z], v)
    uuv = np.cross([x, y, z], uv)
    return v + 2.0 * (w * uv + uuv)


def joint_residuals(phys):
    """||pa - pb|| for every distance-joint, in the sim frame."""
    s = phys.solver
    pos, quat = read_state(phys)
    res = []
    for k in range(s.num_joints):
        a, b = s._j_a[k], s._j_b[k]
        aa = np.asarray(s._j_anchor_a[k], float)
        ab = np.asarray(s._j_anchor_b[k], float)
        pa = pos[a] + quat_rotate(quat[a], aa)
        pb = pos[b] + quat_rotate(quat[b], ab)
        res.append(np.linalg.norm(pa - pb))
    return np.asarray(res)


def kinetic_energy(phys):
    s = phys.solver
    v = s.velocities()
    w = s.omega.numpy().reshape(-1, 3)
    inv_m = np.asarray(s._inv_mass, float)
    ke = 0.0
    for i in range(s.num_bodies):
        if inv_m[i] <= 0.0:
            continue
        m = 1.0 / inv_m[i]
        ke += 0.5 * m * float(np.dot(v[i], v[i]))
        # rough rotational term using body inv_I (diagonal); fine for a trend
        invI = np.asarray(s._inv_I[i], float)
        I = np.where(invI > 0, 1.0 / np.maximum(invI, 1e-12), 0.0)
        wb = quat_rotate([-q for q in s.q.numpy()[i][:3]] + [s.q.numpy()[i][3]], w[i])
        ke += 0.5 * float(np.dot(I, wb * wb))
    return ke


def count_self_pairs(phys):
    """Body-body candidate pairs that survive the solver's full broad-phase
    filter (static-static dropped, same-group>0 dropped, category/mask excluded)."""
    s = phys.solver
    grp = np.asarray(s._group, np.int32)
    cat = np.asarray(s._cat, np.uint64)
    mask = np.asarray(s._mask, np.uint64)
    inv_m = np.asarray(s._inv_mass, float)
    n = s.num_bodies
    cnt = 0
    blocked = 0
    for i in range(n):
        for j in range(i + 1, n):
            if inv_m[i] <= 0 and inv_m[j] <= 0:
                continue
            if grp[i] == grp[j] and grp[i] > 0:
                blocked += 1
                continue
            if (cat[i] & mask[j]) == np.uint64(0) or (cat[j] & mask[i]) == np.uint64(0):
                blocked += 1                 # excluded by the bitmask (adjacent links)
                continue
            cnt += 1
    return cnt, blocked


def _seg_seg_dist(p1, q1, p2, q2):
    """Closest distance between segments [p1,q1] and [p2,q2] (Ericson §5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a <= 1e-12 and e <= 1e-12:
        return np.linalg.norm(p1 - p2)
    if a <= 1e-12:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e <= 1e-12:
            t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0, 1) if denom > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1:
                t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t))


def real_penetration(phys):
    """Actual surface penetration between *collidable* (filter-passing) capsule/
    cylinder proxy pairs — segment+radius, exactly as the solver's narrow phase
    sees them. The honest "are links interpenetrating" number (box-involved pairs
    skipped). Returns (n_penetrating, worst_depth_cm)."""
    s = phys.solver
    pos, quat = read_state(phys)
    st = np.asarray(s._shape_type, int)
    rad = np.asarray(s._shape_r, float)
    hl = np.asarray(s._shape_hl, float)
    cat = np.asarray(s._cat, np.uint64)
    mask = np.asarray(s._mask, np.uint64)
    n = s.num_bodies
    npen, worst = 0, 0.0
    for i in range(n):
        if st[i] not in (1, 2):
            continue
        for j in range(i + 1, n):
            if st[j] not in (1, 2):
                continue
            if (cat[i] & mask[j]) == np.uint64(0) or (cat[j] & mask[i]) == np.uint64(0):
                continue                          # excluded pair (not collidable)
            axi = quat_rotate(quat[i], [0, 0, 1.0])
            axj = quat_rotate(quat[j], [0, 0, 1.0])
            d = _seg_seg_dist(pos[i] - axi * hl[i], pos[i] + axi * hl[i],
                              pos[j] - axj * hl[j], pos[j] + axj * hl[j])
            pen = (rad[i] + rad[j]) - d
            if pen > 0.005:                       # >5 mm true surface overlap
                npen += 1
                worst = max(worst, pen)
    return npen, worst


def active_contacts(phys):
    """Ground-truth body-body contacts the solver actually generated last substep
    (all shape types): number of contact pairs and total manifold points. Proves
    self-collision is doing work, independent of any proxy approximation here."""
    s = phys.solver
    if s.m_count is None or s.n_pairs == 0:
        return 0, 0
    mc = s.m_count.numpy()[:s.n_pairs]
    return int((mc > 0).sum()), int(mc.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--drop", action="store_true",
                   help="base free (collapse on floor) instead of pinned hang")
    p.add_argument("--frames", type=int, default=400)
    p.add_argument("--substeps", type=int, default=20)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--friction", type=float, default=0.8)
    p.add_argument("--lin-damp", type=float, default=0.0)
    p.add_argument("--ang-damp", type=float, default=0.0)
    p.add_argument("--hinge-delta", type=float, default=0.06)
    p.add_argument("--no-self-collision", action="store_true",
                   help="disable per-pair self-collision (old single-group behaviour)")
    args = p.parse_args()

    model = load_mjcf(MJCF)
    q0 = home_pose(model)
    phys = build_xpbd(
        model, q0=q0, base_static=not args.drop, device=args.device,
        substeps=args.substeps, iterations=args.iterations, friction=args.friction,
        lin_damp=args.lin_damp, ang_damp=args.ang_damp, hinge_delta=args.hinge_delta,
        self_collision=not args.no_self_collision,
        start_clearance=0.12 if args.drop else 0.06)
    s = phys.solver
    s._flush()

    npairs, blocked = count_self_pairs(phys)
    print(f"[setup] bodies={s.num_bodies}  joints={s.num_joints}  "
          f"groups={sorted(set(s._group))}")
    print(f"[self-collision] live body-body pairs after filter = {npairs}   "
          f"(blocked by same-group filter = {blocked})")
    np0, w0 = real_penetration(phys)
    print(f"[self-collision] real collidable-pair penetration at t=0: {np0} pairs "
          f"(worst {w0*100:.1f} cm)")
    print(f"[damping] lin_damp={args.lin_damp} ang_damp={args.ang_damp} "
          f"hinge_delta={args.hinge_delta}")
    print()
    print(f"{'frame':>6} {'maxV':>8} {'maxW':>8} {'KE':>10} "
          f"{'jres(mm)':>9} {'penetr':>7} {'contacts':>9} {'pts':>5}")

    recaptures = -1               # first capture isn't a "recapture"
    last_graph = object()
    cap0 = s.cap
    for f in range(args.frames):
        s.step()
        if id(s._graph) != id(last_graph):
            recaptures += 1
            last_graph = s._graph
        if f % 25 == 0 or f == args.frames - 1:
            v = s.velocities()
            w = s.omega.numpy().reshape(-1, 3)
            inv_m = np.asarray(s._inv_mass, float)
            dyn = inv_m > 0
            maxv = float(np.linalg.norm(v[dyn], axis=1).max()) if dyn.any() else 0.0
            maxw = float(np.linalg.norm(w[dyn], axis=1).max()) if dyn.any() else 0.0
            ke = kinetic_energy(phys)
            res = joint_residuals(phys)
            npen, _ = real_penetration(phys)
            nc, npts = active_contacts(phys)
            print(f"{f:>6} {maxv:>8.3f} {maxw:>8.3f} {ke:>10.4f} "
                  f"{res.max()*1000:>9.3f} {npen:>7} {nc:>9} {npts:>5}")

    print()
    res = joint_residuals(phys)
    print(f"[final] joint residual max={res.max()*1000:.3f} mm  "
          f"mean={res.mean()*1000:.3f} mm")
    print(f"[final] finite state: {np.isfinite(s.positions()).all()}")
    print(f"[gpu] graph recaptures over {args.frames} frames: {recaptures}  "
          f"(cap {cap0} -> {s.cap})")


if __name__ == "__main__":
    main()
