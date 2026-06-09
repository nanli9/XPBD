"""6-DOF rigid-body XPBD solver (NVIDIA Warp).

Implements Müller et al. 2020 "Detailed Rigid Body Simulation with Extended
Position Based Dynamics" (Algorithm 2) — see
``reference/Mueller2020_RigidBodyXPBD.pdf``. Rigid **boxes** with quaternion
orientation, rotated inertia, corner-vs-OBB contacts (floor + box-box),
position-level static friction and a velocity pass for dynamic friction +
restitution. Contacts are solved with the paper's order-independent **Jacobi**
projection (atomic accumulation + averaged apply) — fully parallel on the GPU,
no per-frame graph coloring.

Mirrors the AVBD ``Solver6DOF`` role: the same kind of scene (stacks, dominoes)
runs on CUDA with live viewer knobs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from . import kernels_6dof as K6
from .scene import Shape
from .solver import _grid_candidate_pairs


@dataclass
class RigidBody:
    index: int
    half_extents: tuple[float, float, float]
    color: tuple[float, float, float] = (0.6, 0.6, 0.7)
    static: bool = False


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
        self.bodies: list[RigidBody] = []
        self._dirty = True

        # warp arrays
        self.x = self.q = self.v = self.omega = None
        self.x_prev = self.q_prev = None
        self.inv_mass = self.inv_I = self.he = None
        self.dx = self.drot = self.dcount = None
        self.dv = self.dw = self.dvc = None
        self.lam_floor = None
        self.pair_a = self.pair_b = self.lam_pair = None
        self.n_pairs = 0
        self._cur_margin = 0.0
        # box-box manifold buffers (sized to n_pairs each step)
        self.m_count = self.m_inc = self.m_ref = self.m_normal = None
        self.m_off_inc = self.m_off_ref = self.poly = None

    # ---- scene building -----------------------------------------------------
    def add_box(self, position, half_extents, mass,
                quaternion=(0.0, 0.0, 0.0, 1.0), velocity=(0.0, 0.0, 0.0),
                omega=(0.0, 0.0, 0.0), color=(0.6, 0.6, 0.75), static=False):
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
        else:
            self._inv_mass.append(1.0 / float(mass))
            self._inv_I.append(_box_inv_inertia(he, float(mass)))
        self._he.append(he)
        self._dirty = True
        rb = RigidBody(index=idx, half_extents=he, color=color, static=static)
        self.bodies.append(rb)
        return rb

    # ---- broad phase --------------------------------------------------------
    def _rebuild_pairs(self):
        pos = self.x.numpy().reshape(-1, 3)
        he = np.asarray(self._he, dtype=np.float32).reshape(-1, 3)
        inv_m = np.asarray(self._inv_mass, dtype=np.float32)
        n = len(pos)
        if n < 2:
            self.n_pairs = 0
            return np.zeros(0, np.int32), np.zeros(0, np.int32)
        # Tight world-AABB bound (|R|·he per axis) instead of the bounding
        # sphere — for boxes the sphere radius (the diagonal) is ~√3× the
        # half-extent, which inflates the grid cell and over-generates
        # candidates. The AABB keeps the cell ~2× smaller → far fewer pairs.
        aabb = _world_aabb_half(self.q.numpy().reshape(-1, 4), he)   # (n, 3)
        vmax = float(np.abs(self.v.numpy()).max()) if self.v is not None else 0.0
        wmax = float(np.abs(self.omega.numpy()).max()) if self.omega is not None else 0.0
        rmax = float(aabb.max())
        # margin floor = manifold gap, so resting contacts (gap < man_margin)
        # stay paired even when the velocity-aware term vanishes at rest.
        maxhe = float(np.linalg.norm(he, axis=1).max())
        margin = min(self._man_margin + 2.0 * self.dt * (vmax + wmax * maxhe), rmax)
        self._cur_margin = float(margin)
        cell = 2.0 * rmax + margin
        ga, gb = _grid_candidate_pairs(pos, cell)
        if len(ga) == 0:
            self.n_pairs = 0
            return np.zeros(0, np.int32), np.zeros(0, np.int32)
        # exact AABB-overlap filter (per axis), padded by margin
        sep = np.abs(pos[ga] - pos[gb]) - (aabb[ga] + aabb[gb] + margin)
        keep = np.all(sep <= 0.0, axis=1)
        a = ga[keep].astype(np.int32); b = gb[keep].astype(np.int32)
        live = (inv_m[a] > 0.0) | (inv_m[b] > 0.0)   # drop static-static
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
        self.dx = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.drot = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.dcount = wp.zeros(n, dtype=int, device=dev)
        self.dv = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.dw = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.dvc = wp.zeros(n, dtype=int, device=dev)
        self.lam_floor = wp.zeros(n * 8, dtype=float, device=dev)
        # Fixed-capacity contact-pair buffers so the per-substep launch dims are
        # constant — a prerequisite for CUDA-graph capture of the hot loop. Each
        # box in a dense pile touches ~6-7 neighbours; size generously and grow
        # (re-capturing the graph) only if a frame ever exceeds it.
        # With the tight AABB broad phase a dense pile yields ≈ n contact pairs
        # (mostly the vertical stacking contacts), so n·4 is ample headroom;
        # step() grows + re-captures if a frame ever exceeds it. A smaller cap
        # means the box kernels (launched at `cap`) schedule far fewer gated
        # no-op threads.
        self.cap = max(64, n * 4)
        self._alloc_pair_buffers(self.cap, dev)
        self.n_pairs_dev = wp.zeros(1, dtype=int, device=dev)
        self._pa_host = np.zeros(self.cap, np.int32)
        self._pb_host = np.zeros(self.cap, np.int32)
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
            wp.launch(K6.generate_box_manifold, dim=cap,
                      inputs=[self.x, self.q, self.he, self.pair_a, self.pair_b,
                              self.n_pairs_dev, man_margin, self.poly],
                      outputs=[self.m_count, self.m_inc, self.m_ref, self.m_normal,
                               self.m_off_inc, self.m_off_ref], device=dev)
            for _it in range(self.iterations):
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

    def _signature(self, n):
        return (n, self.cap, self.substeps, self.iterations, self.gyroscopic,
                round(self.dt, 9), self.gravity, round(self.floor_y, 6),
                round(self.friction, 6), round(self.restitution, 6),
                round(self.lin_damp, 6), round(self.ang_damp, 6),
                round(self.max_speed, 4), round(self.max_omega, 4),
                round(self._man_margin, 6))

    def step(self):
        self._flush()
        n = len(self._x)
        if n == 0:
            return
        dev = self.device
        # broad phase (host) → write pairs + count into the fixed-capacity
        # device buffers, growing capacity (and dropping the graph) if needed.
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
