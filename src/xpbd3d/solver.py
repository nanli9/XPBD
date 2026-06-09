"""XPBD 3D particle solver (NVIDIA Warp).

A faithful, GPU-resident implementation of Macklin et al. 2016 "XPBD:
Position-Based Simulation of Compliant Constrained Dynamics" (Algorithm 1) with
small-substep integration (Müller et al. 2020). Mirrors the structure of the
sibling AVBD 3D solver so the two share a viewer idiom.

Two equivalent solve modes (both straight from the paper):

* ``"gs"``     — colored Gauss-Seidel. Constraints are graph-colored so a color
                 class touches disjoint particles; one launch per color, fully
                 parallel within a color, sequential across colors. Best
                 convergence per iteration. Coloring is recomputed only when the
                 constraint set changes.
* ``"jacobi"`` — the XPBD paper's 3D GPU mode (§6): every constraint projected in
                 one launch, corrections accumulated with atomics and averaged.
                 No coloring needed, so dynamic contact sets cost nothing extra
                 to maintain. Default on CUDA.

Hard pins are modelled the robust way — a body with ``mass <= 0`` has
``inv_mass = 0`` and never moves (``integrate`` / the solve kernels early-out),
so a chain hung from a static particle stays attached without any constraint.
``ATTACH`` is the *compliant* pin (a spring to a world point) for soft anchors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import warp as wp

from . import kernels as K
from .coloring import color_constraints, color_ranges, color_summary
from .scene import Body, ConstraintHandle, Shape

# Constraint type codes (must match kernels.py)
DISTANCE = 0
ATTACH = 1
FLOOR = 2
CONTACT = 3


@dataclass
class _Row:
    type: int
    body_a: int
    body_b: int = -1
    rest: float = 0.0
    compliance: float = 0.0
    anchor: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mu: float = 0.0


class Solver:
    """3D XPBD particle solver.

    Args:
        dt: frame time step (seconds). Each ``step()`` advances by ``dt``.
        substeps: integration sub-steps per frame (Müller 2020 "small steps").
            More substeps ⇒ stiffer/more stable behaviour at fixed total work.
        iterations: constraint sweeps per substep (Algorithm 1 inner loop).
        gravity: external acceleration (m/s²).
        damping: per-substep viscous velocity damping in [0, 1).
        solve_mode: ``"gs"``, ``"jacobi"``, or ``"auto"`` (Jacobi on CUDA,
            Gauss-Seidel on CPU).
        jacobi_relax: under-relaxation factor for the averaged Jacobi apply.
        device: Warp device, e.g. ``"cpu"`` or ``"cuda:0"``.
        max_speed: velocity-magnitude clamp (runaway guard); 0 disables.
    """

    def __init__(
        self,
        dt: float = 1.0 / 60.0,
        substeps: int = 10,
        iterations: int = 1,
        gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
        damping: float = 0.0,
        solve_mode: str = "auto",
        jacobi_relax: float = 1.0,
        device: str = "cpu",
        max_speed: float = 40.0,
    ):
        wp.init()
        self.device = device
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.iterations = int(iterations)
        self.gravity = tuple(float(g) for g in gravity)
        self.damping = float(damping)
        self.jacobi_relax = float(jacobi_relax)
        self.max_speed = float(max_speed)
        if solve_mode == "auto":
            solve_mode = "jacobi" if str(device).startswith("cuda") else "gs"
        assert solve_mode in ("gs", "jacobi")
        self.solve_mode = solve_mode

        # --- host-side scene description ---
        self._x: list[tuple[float, float, float]] = []
        self._v: list[tuple[float, float, float]] = []
        self._mass: list[float] = []
        self._radius: list[float] = []       # collision radius (0 = no broad phase)
        self._collide: list[bool] = []
        self._friction: list[float] = []
        self._rows: list[_Row] = []

        # Self-collision (sphere-sphere) dynamic pool.
        self._self_collision = False
        self._contact_margin = 0.0
        self._default_friction = 0.0
        self._contact_pool_start: int | None = None

        self._dirty = True

        # --- Warp arrays (built lazily in _flush) ---
        self.x = self.v = self.x_prev = self.inv_mass = None
        self.c_type = self.c_body_a = self.c_body_b = None
        self.c_rest = self.c_compliance = self.c_anchor = self.c_mu = None
        self.c_lambda = self.c_active = None
        self.color_order = None
        self._color_starts = np.zeros(1, dtype=np.int32)
        self.num_colors = 0
        self.color_counts: dict[int, int] = {}
        self.dx_accum = self.dn_accum = None

    # ---- scene building -----------------------------------------------------

    def add_particle(
        self,
        position: tuple[float, float, float],
        mass: float,
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        shape: Shape | None = None,
        collide: bool = True,
        radius: float | None = None,
        friction: float | None = None,
    ) -> Body:
        idx = len(self._x)
        self._x.append(tuple(float(v) for v in position))
        self._v.append(tuple(float(v) for v in velocity))
        self._mass.append(float(mass))
        sh = shape or Shape()
        r = float(radius) if radius is not None else _shape_radius(sh)
        self._radius.append(r)
        self._collide.append(bool(collide) and r > 0.0 and mass > 0.0)
        mu = float(friction) if friction is not None else self._default_friction
        self._friction.append(max(0.0, mu))
        self._dirty = True
        return Body(index=idx, shape=sh)

    def add_distance(
        self,
        body_a: Body,
        body_b: Body,
        rest: float | None = None,
        compliance: float = 0.0,
    ) -> ConstraintHandle:
        """Compliant distance constraint ``‖x_a − x_b‖ = rest``.

        ``compliance`` is the XPBD inverse stiffness α (m/N); 0 = infinitely
        stiff. If ``rest`` is ``None`` it is taken from the current separation.
        """
        if rest is None:
            pa = np.asarray(self._x[body_a.index])
            pb = np.asarray(self._x[body_b.index])
            rest = float(np.linalg.norm(pa - pb))
        idx = len(self._rows)
        self._rows.append(_Row(DISTANCE, body_a.index, body_b.index,
                               rest=float(rest), compliance=float(compliance)))
        self._dirty = True
        return ConstraintHandle(index=idx)

    def add_attach(
        self,
        body: Body,
        world_point: tuple[float, float, float],
        compliance: float = 0.0,
        rest: float = 0.0,
    ) -> ConstraintHandle:
        """Compliant pin: spring the particle toward ``world_point``.

        For a *hard* pin prefer ``add_particle(mass=0)`` (a static body) — that
        is exact and singularity-free. This compliant attach is the soft spring
        the XPBD paper uses to show stiffness control."""
        idx = len(self._rows)
        self._rows.append(_Row(ATTACH, body.index, -1, rest=float(rest),
                               compliance=float(compliance),
                               anchor=tuple(float(p) for p in world_point)))
        self._dirty = True
        return ConstraintHandle(index=idx)

    def add_floor_contact(
        self,
        body: Body,
        floor_y: float = 0.0,
        friction: float | None = None,
    ) -> ConstraintHandle:
        """One-sided floor at ``y = floor_y`` (push-only) with Coulomb friction."""
        mu = float(friction) if friction is not None else self._friction[body.index]
        idx = len(self._rows)
        self._rows.append(_Row(FLOOR, body.index, -1,
                               anchor=(0.0, float(floor_y), 0.0), mu=max(0.0, mu)))
        self._dirty = True
        return ConstraintHandle(index=idx)

    def enable_self_collision(
        self,
        enabled: bool = True,
        margin: float = 0.0,
        default_friction: float | None = None,
    ) -> None:
        """Turn on per-frame broad-phase sphere-sphere contact generation.

        Each ``step()`` scans bodies whose ``collide=True`` for overlapping
        pairs and emits one ``CONTACT`` row each, discarded after the step.
        Pairs already joined by a constraint (chain links) are skipped.
        """
        self._self_collision = bool(enabled)
        self._contact_margin = float(margin)
        if default_friction is not None:
            self._default_friction = max(0.0, float(default_friction))
            for k in range(len(self._friction)):
                if self._friction[k] == 0.0:
                    self._friction[k] = self._default_friction
        if enabled and self._contact_pool_start is None:
            self._contact_pool_start = len(self._rows)

    # ---- runtime perturbations (interactive demos) --------------------------

    def set_position(self, body: Body, p: tuple[float, float, float]) -> None:
        self._flush()
        xs = self.x.numpy().copy()
        xs[body.index] = np.asarray(p, dtype=np.float32)
        self.x = wp.array(xs, dtype=wp.vec3, device=self.device)

    def set_velocity(self, body: Body, vel: tuple[float, float, float]) -> None:
        self._flush()
        vs = self.v.numpy().copy()
        vs[body.index] = np.asarray(vel, dtype=np.float32)
        self.v = wp.array(vs, dtype=wp.vec3, device=self.device)

    def add_impulse(self, body: Body, dv: tuple[float, float, float]) -> None:
        self._flush()
        vs = self.v.numpy().copy()
        vs[body.index] += np.asarray(dv, dtype=np.float32)
        self.v = wp.array(vs, dtype=wp.vec3, device=self.device)

    def set_all_compliance(self, value: float, only_distance: bool = True) -> None:
        """Live-update compliance for every (distance) constraint — used by the
        viewer's stiffness slider to demonstrate iteration-independent stiffness."""
        self._flush()
        comp = self.c_compliance.numpy().copy()
        for i, r in enumerate(self._rows):
            if (not only_distance) or r.type == DISTANCE:
                comp[i] = float(value)
                r.compliance = float(value)
        self.c_compliance = wp.array(comp, dtype=float, device=self.device)

    def set_all_friction(self, mu: float) -> None:
        self._flush()
        self._default_friction = float(mu)
        for k in range(len(self._friction)):
            self._friction[k] = float(mu)
        fr = self.c_mu.numpy().copy()
        for i, r in enumerate(self._rows):
            if r.type in (FLOOR, CONTACT):
                r.mu = float(mu)
                fr[i] = float(mu)
        self.c_mu = wp.array(fr, dtype=float, device=self.device)

    # ---- broad phase --------------------------------------------------------

    def _rebuild_contacts(self) -> None:
        """Drop last frame's CONTACT rows; regenerate from current positions.

        Vectorised O(N²) coarse pass over collidable bodies. Pairs already
        joined by any active constraint are skipped."""
        if not self._self_collision:
            return
        # Strip previous CONTACT rows, keep everything else.
        self._rows = [r for r in self._rows if r.type != CONTACT]
        self._contact_pool_start = len(self._rows)

        # Pairs already constrained (so chain links don't double up).
        existing: set[tuple[int, int]] = set()
        for r in self._rows:
            if r.body_b >= 0:
                a, b = (r.body_a, r.body_b) if r.body_a < r.body_b else (r.body_b, r.body_a)
                existing.add((a, b))

        pos = (self.x.numpy().reshape(-1, 3) if self.x is not None
               else np.asarray(self._x, dtype=np.float32).reshape(-1, 3))
        collide = np.asarray(self._collide, dtype=bool)
        idx = np.nonzero(collide)[0]
        if len(idx) < 2:
            self._dirty = True
            return
        radii = np.asarray(self._radius, dtype=np.float32)
        friction = np.asarray(self._friction, dtype=np.float32)
        # Velocity-aware safety margin (Müller et al. 2020 §3.4): contacts are
        # rebuilt once per frame but solved across all substeps, so a pair must
        # already have a constraint *before* it touches — otherwise a fast body
        # crosses the contact threshold mid-frame, penetrates freely, and gets
        # an explosive separation impulse next frame. We pad the trigger radius
        # by how far the fastest body can travel in one frame (the constraint
        # itself stays inactive — C ≥ 0 — until the bodies actually overlap).
        vmax = (float(np.abs(self.v.numpy()).max()) if self.v is not None else 0.0)
        rmax = float(radii[idx].max())
        # Cap the margin at one body radius: a larger pad would only matter for
        # bodies moving faster than ~radius/frame (rare, and the max-speed clamp
        # already bounds that), while letting a transient velocity spike inflate
        # the grid cell size — which would collapse the O(N) grid back to O(N²)
        # by piling every body into a handful of giant cells.
        margin = min(self._contact_margin + 3.0 * self.dt * vmax, rmax)

        # Uniform spatial-hash grid broad phase (O(N) for near-uniform density).
        # Cell side = max collision diameter + margin, so any overlapping pair
        # lands in the same or an adjacent cell. We gather candidate pairs from
        # each cell's 13 forward neighbours + its own interior, then do the
        # exact distance test vectorised. This replaces the dense O(N²) sweep
        # that dominated the step at scale (≈50 ms → ≈3 ms at 1.1k bodies).
        ga, gb = _grid_candidate_pairs(pos[idx], float(radii[idx].max()) * 2.0 + margin)
        if len(ga) == 0:
            self._dirty = True
            return
        a_glob = idx[ga]; b_glob = idx[gb]
        d = pos[a_glob] - pos[b_glob]
        d2 = np.einsum("ij,ij->i", d, d)
        rs = radii[a_glob] + radii[b_glob] + margin
        keep = d2 <= rs * rs
        a_glob = a_glob[keep]; b_glob = b_glob[keep]
        mus = np.sqrt(friction[a_glob] * friction[b_glob])
        rest = radii[a_glob] + radii[b_glob]
        for a, b, mu, rst in zip(a_glob.tolist(), b_glob.tolist(),
                                 mus.tolist(), rest.tolist()):
            key = (a, b) if a < b else (b, a)
            if key in existing:
                continue
            self._rows.append(_Row(CONTACT, a, b, rest=float(rst), mu=float(mu)))
        self._dirty = True

    # ---- flush: host lists → Warp arrays ------------------------------------

    def _flush(self) -> None:
        if not self._dirty:
            return
        dev = self.device
        n_b = len(self._x)
        n_c = len(self._rows)

        # Preserve already-simulated body state across a mid-sim flush.
        cur_x = self.x.numpy() if self.x is not None else None
        cur_v = self.v.numpy() if self.v is not None else None
        n_b_prev = 0 if cur_x is None else int(cur_x.shape[0])

        x_np = np.asarray(self._x, dtype=np.float32).reshape(-1, 3) if n_b else np.zeros((0, 3), np.float32)
        v_np = np.asarray(self._v, dtype=np.float32).reshape(-1, 3) if n_b else np.zeros((0, 3), np.float32)
        if 0 < n_b_prev <= n_b:
            x_np[:n_b_prev] = cur_x[:n_b_prev]
            v_np[:n_b_prev] = cur_v[:n_b_prev]

        mass = np.asarray(self._mass, dtype=np.float32) if n_b else np.zeros(0, np.float32)
        inv_mass = np.where(mass > 0.0, 1.0 / np.maximum(mass, 1e-20), 0.0).astype(np.float32)

        self.x = wp.array(x_np, dtype=wp.vec3, device=dev)
        self.v = wp.array(v_np, dtype=wp.vec3, device=dev)
        self.x_prev = wp.zeros(n_b, dtype=wp.vec3, device=dev)
        self.inv_mass = wp.array(inv_mass, dtype=float, device=dev)
        self.dx_accum = wp.zeros(n_b, dtype=wp.vec3, device=dev)
        self.dn_accum = wp.zeros(n_b, dtype=int, device=dev)

        def i32(seq):
            return np.asarray(seq, dtype=np.int32) if n_c else np.zeros(0, np.int32)

        def f32(seq):
            return np.asarray(seq, dtype=np.float32) if n_c else np.zeros(0, np.float32)

        anchor = (np.asarray([r.anchor for r in self._rows], dtype=np.float32).reshape(-1, 3)
                  if n_c else np.zeros((0, 3), np.float32))
        c_body_a = i32([r.body_a for r in self._rows])
        c_body_b = i32([r.body_b for r in self._rows])
        self.c_type = wp.array(i32([r.type for r in self._rows]), dtype=int, device=dev)
        self.c_body_a = wp.array(c_body_a, dtype=int, device=dev)
        self.c_body_b = wp.array(c_body_b, dtype=int, device=dev)
        self.c_rest = wp.array(f32([r.rest for r in self._rows]), dtype=float, device=dev)
        self.c_compliance = wp.array(f32([r.compliance for r in self._rows]), dtype=float, device=dev)
        self.c_anchor = wp.array(anchor, dtype=wp.vec3, device=dev)
        self.c_mu = wp.array(f32([r.mu for r in self._rows]), dtype=float, device=dev)
        self.c_lambda = wp.zeros(n_c, dtype=float, device=dev)
        self.c_active = wp.ones(n_c, dtype=int, device=dev)

        # Coloring (only needed for Gauss-Seidel).
        if self.solve_mode == "gs" and n_c > 0:
            color = color_constraints(n_b, c_body_a, c_body_b)
            order, starts, num_colors = color_ranges(color)
            self.color_order = wp.array(order, dtype=int, device=dev)
            self._color_starts = starts
            self.num_colors = num_colors
            self.color_counts = color_summary(color)
        else:
            self.color_order = wp.zeros(max(n_c, 1), dtype=int, device=dev)
            self._color_starts = np.array([0, n_c], dtype=np.int32)
            self.num_colors = 1 if n_c else 0
            self.color_counts = {}

        self._dirty = False

    # ---- the step -----------------------------------------------------------

    def step(self) -> None:
        if self._self_collision:
            self._flush()
            self._sanitize_positions()
            self._rebuild_contacts()
        self._flush()
        n_b = len(self._x)
        n_c = len(self._rows)
        if n_b == 0:
            return
        dev = self.device
        sdt = self.dt / float(self.substeps)
        grav = wp.vec3(*self.gravity)

        for _ in range(self.substeps):
            wp.launch(K.integrate, dim=n_b,
                      inputs=[self.x, self.v, self.inv_mass, grav, sdt],
                      outputs=[self.x_prev], device=dev)
            if n_c > 0:
                self.c_lambda.zero_()  # Alg. 1 line 4: λ₀ ← 0 each substep
                for _it in range(self.iterations):
                    if self.solve_mode == "gs":
                        self._sweep_gs(n_c, sdt, dev)
                    else:
                        self._sweep_jacobi(n_c, n_b, sdt, dev)
            wp.launch(K.finalize_velocity, dim=n_b,
                      inputs=[self.x, self.x_prev, self.inv_mass, sdt, self.damping],
                      outputs=[self.v], device=dev)
            if self.max_speed > 0.0 and math.isfinite(self.max_speed):
                wp.launch(K.cap_velocity, dim=n_b,
                          inputs=[self.v, self.max_speed], device=dev)

    def _sweep_gs(self, n_c, sdt, dev) -> None:
        for c in range(self.num_colors):
            start = int(self._color_starts[c])
            count = int(self._color_starts[c + 1]) - start
            if count <= 0:
                continue
            wp.launch(K.solve_constraints_color, dim=count,
                      inputs=[self.x, self.x_prev, self.inv_mass,
                              self.c_type, self.c_body_a, self.c_body_b,
                              self.c_rest, self.c_compliance, self.c_anchor,
                              self.c_mu, self.c_lambda, self.c_active,
                              self.color_order, start, sdt],
                      device=dev)

    def _sweep_jacobi(self, n_c, n_b, sdt, dev) -> None:
        wp.launch(K.solve_constraints_jacobi, dim=n_c,
                  inputs=[self.x, self.x_prev, self.inv_mass,
                          self.c_type, self.c_body_a, self.c_body_b,
                          self.c_rest, self.c_compliance, self.c_anchor,
                          self.c_mu, self.c_lambda, self.c_active, sdt],
                  outputs=[self.dx_accum, self.dn_accum], device=dev)
        wp.launch(K.apply_jacobi, dim=n_b,
                  inputs=[self.x, self.dx_accum, self.dn_accum,
                          self.inv_mass, self.jacobi_relax], device=dev)

    def _sanitize_positions(self) -> None:
        """Reset any non-finite body positions to their initial scene values
        before the broad phase squares them into a crash."""
        if self.x is None:
            return
        xs = self.x.numpy()
        if np.all(np.isfinite(xs)):
            return
        bad = ~np.all(np.isfinite(xs), axis=1)
        safe = np.asarray(self._x, dtype=np.float32).reshape(-1, 3)
        xs[bad] = safe[bad]
        self.x = wp.array(xs, dtype=wp.vec3, device=self.device)
        vs = self.v.numpy()
        vs[bad] = 0.0
        self.v = wp.array(vs, dtype=wp.vec3, device=self.device)

    # ---- read-back ----------------------------------------------------------

    def positions(self) -> np.ndarray:
        if self.x is None:
            return np.asarray(self._x, dtype=np.float32).reshape(-1, 3)
        return self.x.numpy().reshape(-1, 3)

    def velocities(self) -> np.ndarray:
        if self.v is None:
            return np.asarray(self._v, dtype=np.float32).reshape(-1, 3)
        return self.v.numpy().reshape(-1, 3)

    def lambdas(self) -> np.ndarray:
        if self.c_lambda is None:
            return np.zeros(len(self._rows), dtype=np.float32)
        return self.c_lambda.numpy()

    @property
    def num_bodies(self) -> int:
        return len(self._x)

    @property
    def num_constraints(self) -> int:
        return len(self._rows)


# 13 "forward" neighbour offsets (half of the 26-cell shell, so each ordered
# pair of distinct cells is visited once); the cell's own interior pairs are
# handled separately.
_FWD_OFFSETS = (
    (1, -1, -1), (1, -1, 0), (1, -1, 1), (1, 0, -1), (1, 0, 0), (1, 0, 1),
    (1, 1, -1), (1, 1, 0), (1, 1, 1), (0, 1, -1), (0, 1, 0), (0, 1, 1), (0, 0, 1),
)


def _grid_candidate_pairs(p: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate local-index pairs ``(a, b)`` whose bodies fall in the
    same or an adjacent grid cell of side ``cell``. Exact distance filtering is
    done by the caller. ``p`` is ``(M, 3)`` positions of the collidable bodies."""
    m = p.shape[0]
    if m < 2 or cell <= 0.0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    cells = np.floor(p / cell).astype(np.int64).tolist()
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for k in range(m):
        buckets.setdefault(tuple(cells[k]), []).append(k)
    ca: list[int] = []
    cb: list[int] = []
    for key, mem in buckets.items():
        n = len(mem)
        for ii in range(n):  # interior pairs of this cell
            mi = mem[ii]
            for jj in range(ii + 1, n):
                ca.append(mi); cb.append(mem[jj])
        kx, ky, kz = key
        for ox, oy, oz in _FWD_OFFSETS:  # forward-neighbour pairs
            nb = buckets.get((kx + ox, ky + oy, kz + oz))
            if nb:
                for mi in mem:
                    for mj in nb:
                        ca.append(mi); cb.append(mj)
    return np.asarray(ca, np.int64), np.asarray(cb, np.int64)


def _shape_radius(shape: Shape | None) -> float:
    """Bounding-sphere radius used for the sphere-sphere broad phase."""
    if shape is None:
        return 0.0
    if shape.kind == "sphere":
        return float(shape.size[0])
    if shape.kind == "cube":
        return float(min(shape.size))  # inscribed sphere
    if shape.kind == "pillar":
        return float(shape.size[0])
    return 0.0
