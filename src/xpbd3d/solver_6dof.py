"""6-DOF rigid-body XPBD solver (NVIDIA Warp).

Implements Müller et al. 2020 "Detailed Rigid Body Simulation with Extended
Position Based Dynamics" (Algorithm 2) — see
``reference/Mueller2020_RigidBodyXPBD.pdf``. Rigid **boxes** with quaternion
orientation, rotated inertia, OBB contacts (floor + box-box via SAT + face
clip), position-level static friction and a velocity pass for dynamic friction +
restitution. Contacts are solved with the paper's order-independent **Jacobi**
projection (atomic accumulation + averaged apply) — fully parallel on the GPU,
no per-frame graph coloring.

Beyond rigid boxes the solver also handles **particles** (point masses with no
rotational inertia — used as cloth nodes; ``add_particle``) and **joints**
(compliant XPBD distance/attachment constraints between local anchors on any two
bodies — ``add_joint``), so rigid bodies, cloth and articulated links can share
one substep loop / one captured graph. This mirrors the unified treatment in
Müller 2020 §3.3.

Broad phase is a GPU **LBVH** (``kernels_bvh``): per-body world AABBs are built
and queried entirely on device, emitting candidate pairs with an atomic counter —
no host round-trip. (``broadphase="grid"`` keeps the legacy NumPy spatial hash.)

Mirrors the AVBD ``Solver6DOF`` role: the same kind of scene (stacks, dominoes,
cloth) runs on CUDA with live viewer knobs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from . import kernels_6dof as K6
from . import kernels_bvh as KB
from .solver import _grid_candidate_pairs


@dataclass
class RigidBody:
    index: int
    half_extents: tuple[float, float, float]
    color: tuple[float, float, float] = (0.6, 0.6, 0.7)
    static: bool = False
    is_particle: bool = False
    group: int = 0


def _world_aabb_half(q, he):
    """Vectorised world-AABB half-extents of OBBs: ``|R|·he`` per box.
    ``q`` is (n,4) xyzw quaternions, ``he`` is (n,3) half-extents."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    # |R| columns (abs of the rotation matrix), then |R| · he.
    r00 = np.abs(1 - 2 * (yy + zz)); r01 = np.abs(2 * (xy - wz)); r02 = np.abs(2 * (xz + wy))
    r10 = np.abs(2 * (xy + wz)); r11 = np.abs(1 - 2 * (xx + zz)); r12 = np.abs(2 * (yz - wx))
    r20 = np.abs(2 * (xz - wy)); r21 = np.abs(2 * (yz + wx)); r22 = np.abs(1 - 2 * (xx + yy))
    ax = r00 * he[:, 0] + r01 * he[:, 1] + r02 * he[:, 2]
    ay = r10 * he[:, 0] + r11 * he[:, 1] + r12 * he[:, 2]
    az = r20 * he[:, 0] + r21 * he[:, 1] + r22 * he[:, 2]
    return np.stack([ax, ay, az], axis=1).astype(np.float32)


def _box_inv_inertia(he, mass):
    """Diagonal inverse inertia (body frame) for a solid box."""
    lx, ly, lz = 2.0 * he[0], 2.0 * he[1], 2.0 * he[2]
    ix = mass / 12.0 * (ly * ly + lz * lz)
    iy = mass / 12.0 * (lx * lx + lz * lz)
    iz = mass / 12.0 * (lx * lx + ly * ly)
    return (1.0 / ix, 1.0 / iy, 1.0 / iz)


class Solver6DOF:
    def __init__(
        self,
        dt: float = 1.0 / 60.0,
        substeps: int = 15,
        iterations: int = 1,
        gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
        floor_y: float = 0.0,
        friction: float = 0.6,
        restitution: float = 0.0,
        lin_damp: float = 0.0,
        ang_damp: float = 0.0,
        gyroscopic: bool = False,
        jacobi_relax: float = 1.0,
        device: str = "cuda:0",
        max_speed: float = 60.0,
        max_omega: float = 60.0,
        broadphase: str = "lbvh",
    ):
        wp.init()
        self.device = device
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.iterations = int(iterations)
        self.gravity = tuple(float(g) for g in gravity)
        self.floor_y = float(floor_y)
        self.friction = float(friction)
        self.restitution = float(restitution)
        self.lin_damp = float(lin_damp)
        self.ang_damp = float(ang_damp)
        self.gyroscopic = bool(gyroscopic)
        self.jacobi_relax = float(jacobi_relax)
        self.max_speed = float(max_speed)
        self.max_omega = float(max_omega)
        self.broadphase = str(broadphase)
        self.contact_margin = 0.0
        # Manifold clip gap (fixed, baked into the captured graph). Decoupled
        # from the velocity-aware *broad-phase* margin so it never forces a
        # per-frame graph re-capture. Only governs which about-to-touch clipped
        # points are kept; penetrating contacts always are.
        self._man_margin = 0.02

        # host scene
        self._x: list = []
        self._q: list = []
        self._v: list = []
        self._w: list = []
        self._inv_mass: list = []
        self._inv_I: list = []
        self._he: list = []
        self._group: list = []
        self.bodies: list[RigidBody] = []
        self._dirty = True

        # host joints
        self._j_a: list = []
        self._j_b: list = []
        self._j_anchor_a: list = []
        self._j_anchor_b: list = []
        self._j_rest: list = []
        self._j_compliance: list = []

        # warp arrays
        self.x = self.q = self.v = self.omega = None
        self.x_prev = self.q_prev = None
        self.inv_mass = self.inv_I = self.he = self.cgroup = None
        self.dx = self.drot = self.dcount = None
        self.dv = self.dw = self.dvc = None
        self.lam_floor = None
        self.pair_a = self.pair_b = self.lam_pair = None
        self.n_pairs = 0
        self._cur_margin = 0.0
        # box-box manifold buffers (sized to cap each flush)
        self.m_count = self.m_inc = self.m_ref = self.m_normal = None
        self.m_off_inc = self.m_off_ref = self.poly = None
        # joints
        self.j_a = self.j_b = self.j_anchor_a = self.j_anchor_b = None
        self.j_rest = self.j_compliance = self.lam_joint = None
        self.n_joints = 0
        # GPU LBVH broad phase
        self.lowers = self.uppers = self.bvh = None

    # ---- scene building -----------------------------------------------------
    def _add_body(self, position, half_extents, mass, quaternion, velocity,
                  omega, color, static, is_particle, group):
        idx = len(self._x)
        he = tuple(float(h) for h in half_extents)
        self._x.append(tuple(float(p) for p in position))
        self._q.append(tuple(float(c) for c in quaternion))
        self._v.append(tuple(float(c) for c in velocity))
        self._w.append(tuple(float(c) for c in omega))
        if static or mass <= 0.0:
            self._inv_mass.append(0.0)
            self._inv_I.append((0.0, 0.0, 0.0))
            static = True
        elif is_particle:
            # point mass: no rotational inertia (never spins, behaves as a node)
            self._inv_mass.append(1.0 / float(mass))
            self._inv_I.append((0.0, 0.0, 0.0))
        else:
            self._inv_mass.append(1.0 / float(mass))
            self._inv_I.append(_box_inv_inertia(he, float(mass)))
        self._he.append(he)
        self._group.append(int(group))
        self._dirty = True
        rb = RigidBody(index=idx, half_extents=he, color=color, static=static,
                       is_particle=is_particle, group=int(group))
        self.bodies.append(rb)
        return rb

    def add_box(self, position, half_extents, mass,
                quaternion=(0.0, 0.0, 0.0, 1.0), velocity=(0.0, 0.0, 0.0),
                omega=(0.0, 0.0, 0.0), color=(0.6, 0.6, 0.75), static=False):
        return self._add_body(position, half_extents, mass, quaternion, velocity,
                              omega, color, static, is_particle=False, group=0)

    def add_particle(self, position, mass, radius=0.015,
                     velocity=(0.0, 0.0, 0.0), color=(0.85, 0.3, 0.35),
                     static=False, group=1):
        """A point mass (no rotational inertia). Modelled as a small box of
        half-extent ``radius`` so it shares the box contact path (floor + OBB);
        bodies in the same ``group`` (> 0) don't collide with each other, which
        keeps cloth self-collision off while still colliding with rigid bodies."""
        return self._add_body(position, (radius, radius, radius), mass,
                              (0.0, 0.0, 0.0, 1.0), velocity, (0.0, 0.0, 0.0),
                              color, static, is_particle=True, group=group)

    def add_joint(self, a, b, compliance=0.0, rest_length=None,
                  anchor_a=(0.0, 0.0, 0.0), anchor_b=(0.0, 0.0, 0.0)):
        """Compliant distance/attachment constraint between anchor points fixed
        in the body frames of bodies ``a`` and ``b``. ``rest_length=None`` uses
        the current anchor separation (a rigid link / pin); ``compliance`` is
        XPBD's inverse stiffness (0 = hard, ~1e-6 = stiff cloth edge)."""
        ia = a.index if hasattr(a, "index") else int(a)
        ib = b.index if hasattr(b, "index") else int(b)
        aa = np.asarray(anchor_a, np.float64)
        ab = np.asarray(anchor_b, np.float64)
        if rest_length is None:
            xa = np.asarray(self._x[ia]); xb = np.asarray(self._x[ib])
            rest_length = float(np.linalg.norm((xa + aa) - (xb + ab)))
        self._j_a.append(ia); self._j_b.append(ib)
        self._j_anchor_a.append(tuple(float(c) for c in anchor_a))
        self._j_anchor_b.append(tuple(float(c) for c in anchor_b))
        self._j_rest.append(float(rest_length))
        self._j_compliance.append(float(compliance))
        self._dirty = True
        return len(self._j_a) - 1

    # ---- broad phase (legacy grid; LBVH path is on device in step) ----------
    def _rebuild_pairs(self):
        pos = self.x.numpy().reshape(-1, 3)
        he = np.asarray(self._he, dtype=np.float32).reshape(-1, 3)
        inv_m = np.asarray(self._inv_mass, dtype=np.float32)
        n = len(pos)
        if n < 2:
            self.n_pairs = 0
            return np.zeros(0, np.int32), np.zeros(0, np.int32)
        aabb = _world_aabb_half(self.q.numpy().reshape(-1, 4), he)   # (n, 3)
        vmax = float(np.abs(self.v.numpy()).max()) if self.v is not None else 0.0
        wmax = float(np.abs(self.omega.numpy()).max()) if self.omega is not None else 0.0
        rmax = float(aabb.max())
        maxhe = float(np.linalg.norm(he, axis=1).max())
        margin = min(self._man_margin + 2.0 * self.dt * (vmax + wmax * maxhe), rmax)
        self._cur_margin = float(margin)
        cell = 2.0 * rmax + margin
        ga, gb = _grid_candidate_pairs(pos, cell)
        if len(ga) == 0:
            self.n_pairs = 0
            return np.zeros(0, np.int32), np.zeros(0, np.int32)
        sep = np.abs(pos[ga] - pos[gb]) - (aabb[ga] + aabb[gb] + margin)
        keep = np.all(sep <= 0.0, axis=1)
        a = ga[keep].astype(np.int32); b = gb[keep].astype(np.int32)
        grp = np.asarray(self._group, np.int32)
        live = (inv_m[a] > 0.0) | (inv_m[b] > 0.0)              # drop static-static
        live &= ~((grp[a] == grp[b]) & (grp[a] > 0))           # drop same no-self group
        return a[live], b[live]

    # ---- flush --------------------------------------------------------------
    def _flush(self):
        if not self._dirty:
            return
        dev = self.device
        n = len(self._x)
        cur_x = self.x.numpy() if self.x is not None else None
        cur_q = self.q.numpy() if self.q is not None else None
        cur_v = self.v.numpy() if self.v is not None else None
        cur_w = self.omega.numpy() if self.omega is not None else None
        prev_n = 0 if cur_x is None else int(cur_x.shape[0])

        x = np.asarray(self._x, np.float32).reshape(-1, 3) if n else np.zeros((0, 3), np.float32)
        q = np.asarray(self._q, np.float32).reshape(-1, 4) if n else np.zeros((0, 4), np.float32)
        v = np.asarray(self._v, np.float32).reshape(-1, 3) if n else np.zeros((0, 3), np.float32)
        w = np.asarray(self._w, np.float32).reshape(-1, 3) if n else np.zeros((0, 3), np.float32)
        if 0 < prev_n <= n:
            x[:prev_n] = cur_x[:prev_n]; q[:prev_n] = cur_q[:prev_n]
            v[:prev_n] = cur_v[:prev_n]; w[:prev_n] = cur_w[:prev_n]

        self.x = wp.array(x, dtype=wp.vec3, device=dev)
        self.q = wp.array(q, dtype=wp.quat, device=dev)
        self.v = wp.array(v, dtype=wp.vec3, device=dev)
        self.omega = wp.array(w, dtype=wp.vec3, device=dev)
        self.x_prev = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.q_prev = wp.zeros(n, dtype=wp.quat, device=dev)
        self.inv_mass = wp.array(np.asarray(self._inv_mass, np.float32), dtype=float, device=dev)
        self.inv_I = wp.array(np.asarray(self._inv_I, np.float32).reshape(-1, 3), dtype=wp.vec3, device=dev)
        self.he = wp.array(np.asarray(self._he, np.float32).reshape(-1, 3), dtype=wp.vec3, device=dev)
        self.cgroup = wp.array(np.asarray(self._group, np.int32), dtype=int, device=dev)
        self.dx = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.drot = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.dcount = wp.zeros(n, dtype=int, device=dev)
        self.dv = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.dw = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.dvc = wp.zeros(n, dtype=int, device=dev)
        self.lam_floor = wp.zeros(n * 8, dtype=float, device=dev)

        # joints (fixed topology → live inside the captured graph)
        self.n_joints = len(self._j_a)
        if self.n_joints:
            self.j_a = wp.array(np.asarray(self._j_a, np.int32), dtype=int, device=dev)
            self.j_b = wp.array(np.asarray(self._j_b, np.int32), dtype=int, device=dev)
            self.j_anchor_a = wp.array(np.asarray(self._j_anchor_a, np.float32).reshape(-1, 3), dtype=wp.vec3, device=dev)
            self.j_anchor_b = wp.array(np.asarray(self._j_anchor_b, np.float32).reshape(-1, 3), dtype=wp.vec3, device=dev)
            self.j_rest = wp.array(np.asarray(self._j_rest, np.float32), dtype=float, device=dev)
            self.j_compliance = wp.array(np.asarray(self._j_compliance, np.float32), dtype=float, device=dev)
            self.lam_joint = wp.zeros(self.n_joints, dtype=float, device=dev)

        # Fixed-capacity contact-pair buffers so the per-substep launch dims are
        # constant — a prerequisite for CUDA-graph capture of the hot loop. Each
        # box in a dense pile touches ~6-7 neighbours; size generously and grow
        # (re-capturing the graph) only if a frame ever exceeds it.
        self.cap = max(64, n * 4)
        self._alloc_pair_buffers(self.cap, dev)
        self.n_pairs_dev = wp.zeros(1, dtype=int, device=dev)
        self._pa_host = np.zeros(self.cap, np.int32)
        self._pb_host = np.zeros(self.cap, np.int32)

        # GPU LBVH broad phase: per-body world AABBs + the tree over them.
        self.lowers = wp.zeros(max(n, 1), dtype=wp.vec3, device=dev)
        self.uppers = wp.zeros(max(n, 1), dtype=wp.vec3, device=dev)
        self.bvh = None
        if self.broadphase == "lbvh" and n >= 1:
            wp.launch(KB.compute_body_aabb, dim=n,
                      inputs=[self.x, self.q, self.v, self.omega, self.he,
                              self._man_margin, self.dt],
                      outputs=[self.lowers, self.uppers], device=dev)
            self.bvh = wp.Bvh(self.lowers, self.uppers, constructor="lbvh")

        self._graph = None          # captured CUDA graph (invalidated on change)
        self._graph_sig = None
        self._dirty = False

    def _alloc_pair_buffers(self, cap, dev):
        self.pair_a = wp.zeros(cap, dtype=int, device=dev)
        self.pair_b = wp.zeros(cap, dtype=int, device=dev)
        self.lam_pair = wp.zeros(cap * 8, dtype=float, device=dev)
        self.m_count = wp.zeros(cap, dtype=int, device=dev)
        self.m_inc = wp.zeros(cap, dtype=int, device=dev)
        self.m_ref = wp.zeros(cap, dtype=int, device=dev)
        self.m_normal = wp.zeros(cap, dtype=wp.vec3, device=dev)
        self.m_off_inc = wp.zeros(cap * 8, dtype=wp.vec3, device=dev)
        self.m_off_ref = wp.zeros(cap * 8, dtype=wp.vec3, device=dev)
        self.poly = wp.zeros(cap * 16, dtype=wp.vec3, device=dev)

    # ---- step ---------------------------------------------------------------
    def _record(self, n):
        """Issue all per-substep kernel launches. Used both for direct execution
        (CPU) and for CUDA-graph capture. Box kernels launch at fixed capacity
        ``self.cap`` and gate on the device-side pair count so the launch
        topology is constant across frames (capture-safe)."""
        dev = self.device
        cap = self.cap
        nj = self.n_joints
        h = self.dt / float(self.substeps)
        grav = wp.vec3(*self.gravity)
        mu = self.friction
        man_margin = self._man_margin
        for _ in range(self.substeps):
            wp.launch(K6.integrate_bodies, dim=n,
                      inputs=[self.x, self.q, self.v, self.omega, self.inv_mass,
                              self.inv_I, grav, h, 1 if self.gyroscopic else 0],
                      outputs=[self.x_prev, self.q_prev], device=dev)
            self.lam_floor.zero_()
            self.lam_pair.zero_()
            if nj:
                self.lam_joint.zero_()
            wp.launch(K6.generate_box_manifold, dim=cap,
                      inputs=[self.x, self.q, self.he, self.pair_a, self.pair_b,
                              self.n_pairs_dev, man_margin, self.poly],
                      outputs=[self.m_count, self.m_inc, self.m_ref, self.m_normal,
                               self.m_off_inc, self.m_off_ref], device=dev)
            for _it in range(self.iterations):
                if nj:
                    wp.launch(K6.solve_joints, dim=nj,
                              inputs=[self.x, self.q, self.inv_mass, self.inv_I,
                                      self.j_a, self.j_b, self.j_anchor_a,
                                      self.j_anchor_b, self.j_rest,
                                      self.j_compliance, self.lam_joint, h],
                              outputs=[self.dx, self.drot, self.dcount], device=dev)
                wp.launch(K6.solve_floor_contacts, dim=n,
                          inputs=[self.x, self.q, self.x_prev, self.q_prev,
                                  self.inv_mass, self.inv_I, self.he,
                                  self.floor_y, mu, self.lam_floor, h],
                          outputs=[self.dx, self.drot, self.dcount], device=dev)
                wp.launch(K6.solve_box_manifold, dim=cap,
                          inputs=[self.x, self.q, self.inv_mass, self.inv_I,
                                  self.m_count, self.m_inc, self.m_ref,
                                  self.m_normal, self.m_off_inc, self.m_off_ref,
                                  self.lam_pair],
                          outputs=[self.dx, self.drot, self.dcount], device=dev)
                wp.launch(K6.apply_jacobi_6dof, dim=n,
                          inputs=[self.x, self.q, self.inv_mass, self.jacobi_relax],
                          outputs=[self.dx, self.drot, self.dcount], device=dev)
            wp.launch(K6.finalize_velocity_6dof, dim=n,
                      inputs=[self.x, self.q, self.x_prev, self.q_prev,
                              self.inv_mass, h, self.lin_damp, self.ang_damp],
                      outputs=[self.v, self.omega], device=dev)
            wp.launch(K6.velocity_floor, dim=n,
                      inputs=[self.x, self.q, self.inv_mass, self.inv_I, self.he,
                              self.floor_y, mu, self.restitution, self.lam_floor, h,
                              self.v, self.omega],
                      outputs=[self.dv, self.dw, self.dvc], device=dev)
            wp.launch(K6.velocity_box, dim=cap,
                      inputs=[self.x, self.q, self.inv_mass, self.inv_I,
                              self.m_count, self.m_inc, self.m_ref, self.m_normal,
                              self.m_off_inc, self.m_off_ref, mu, self.restitution,
                              self.lam_pair, h, self.v, self.omega],
                      outputs=[self.dv, self.dw, self.dvc], device=dev)
            wp.launch(K6.apply_velocity_6dof, dim=n,
                      inputs=[self.inv_mass], outputs=[self.dv, self.dw, self.dvc,
                                                       self.v, self.omega], device=dev)
            if self.max_speed > 0.0:
                wp.launch(K6.cap_velocity_6dof, dim=n,
                          inputs=[self.v, self.omega, self.max_speed, self.max_omega],
                          device=dev)

    def _broadphase_gpu(self, n):
        """On-device LBVH broad phase: rebuild per-body AABBs, rebuild the tree,
        emit candidate pairs into the fixed-capacity buffers. Returns the live
        pair count (read back once — a single int — to detect cap overflow)."""
        dev = self.device
        self.n_pairs_dev.zero_()
        wp.launch(KB.compute_body_aabb, dim=n,
                  inputs=[self.x, self.q, self.v, self.omega, self.he,
                          self._man_margin, self.dt],
                  outputs=[self.lowers, self.uppers], device=dev)
        self.bvh.rebuild()
        wp.launch(KB.emit_pairs, dim=n,
                  inputs=[self.bvh.id, self.lowers, self.uppers, self.inv_mass,
                          self.cgroup, self.cap],
                  outputs=[self.n_pairs_dev, self.pair_a, self.pair_b], device=dev)
        cnt = int(self.n_pairs_dev.numpy()[0])
        if cnt > self.cap:                       # overflow → grow + re-emit once
            self.cap = int(cnt * 1.5)
            self._alloc_pair_buffers(self.cap, dev)
            self.n_pairs_dev.zero_()
            wp.launch(KB.emit_pairs, dim=n,
                      inputs=[self.bvh.id, self.lowers, self.uppers, self.inv_mass,
                              self.cgroup, self.cap],
                      outputs=[self.n_pairs_dev, self.pair_a, self.pair_b], device=dev)
            cnt = int(self.n_pairs_dev.numpy()[0])
            self._graph = None
        return cnt

    def _signature(self, n):
        return (n, self.cap, self.n_joints, self.substeps, self.iterations,
                self.gyroscopic, round(self.dt, 9), self.gravity,
                round(self.floor_y, 6), round(self.friction, 6),
                round(self.restitution, 6), round(self.lin_damp, 6),
                round(self.ang_damp, 6), round(self.max_speed, 4),
                round(self.max_omega, 4), round(self._man_margin, 6))

    def step(self):
        self._flush()
        n = len(self._x)
        if n == 0:
            return
        dev = self.device
        use_lbvh = self.broadphase == "lbvh" and self.bvh is not None

        if use_lbvh:
            # all-device broad phase: pairs + count already live in the device
            # buffers the graph reads, so no host pair assignment is needed.
            self.n_pairs = self._broadphase_gpu(n)
        else:
            a, b = self._rebuild_pairs()
            self.n_pairs = int(len(a))
            if self.n_pairs > self.cap:
                self.cap = int(self.n_pairs * 1.5)
                self._alloc_pair_buffers(self.cap, dev)
                self._pa_host = np.zeros(self.cap, np.int32)
                self._pb_host = np.zeros(self.cap, np.int32)
                self._graph = None
            self._pa_host[:self.n_pairs] = a
            self._pb_host[:self.n_pairs] = b
            self.pair_a.assign(self._pa_host)
            self.pair_b.assign(self._pb_host)
            self.n_pairs_dev.assign(np.array([self.n_pairs], np.int32))

        use_graph = str(dev).startswith("cuda")
        if use_graph:
            sig = self._signature(n)
            if self._graph is None or self._graph_sig != sig:
                with wp.ScopedCapture(device=dev) as capture:
                    self._record(n)
                self._graph = capture.graph
                self._graph_sig = sig
            wp.capture_launch(self._graph)
        else:
            self._record(n)

    # ---- perturbation / readback -------------------------------------------
    def set_velocity(self, idx, vel):
        self._flush()
        a = self.v.numpy().copy(); a[idx] = np.asarray(vel, np.float32)
        self.v = wp.array(a, dtype=wp.vec3, device=self.device)
        self._graph = None  # array reassigned → captured graph is stale

    def add_impulse(self, idx, dvel):
        self._flush()
        a = self.v.numpy().copy(); a[idx] += np.asarray(dvel, np.float32)
        self.v = wp.array(a, dtype=wp.vec3, device=self.device)
        self._graph = None

    def positions(self):
        return self.x.numpy().reshape(-1, 3) if self.x is not None else np.asarray(self._x, np.float32).reshape(-1, 3)

    def orientations(self):
        return self.q.numpy().reshape(-1, 4) if self.q is not None else np.asarray(self._q, np.float32).reshape(-1, 4)

    def velocities(self):
        return self.v.numpy().reshape(-1, 3) if self.v is not None else np.asarray(self._v, np.float32).reshape(-1, 3)

    @property
    def num_bodies(self):
        return len(self._x)

    @property
    def num_joints(self):
        return len(self._j_a)
