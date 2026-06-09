"""Interactive 3D viewer for the 6-DOF rigid-body XPBD solver (viser).

The 6-DOF counterpart of ``viewer.py``: real rigid **boxes** with quaternion
orientation, rotated inertia and OBB contacts (Müller et al. 2020), running on
the GPU with a CUDA-graph-captured substep loop. Mirrors the AVBD ``Solver6DOF``
viewer — stacks, dominoes, big block piles, and a unified rigid+cloth+joints
scene.

    uv run python examples/viewer_6dof.py --scene stack
    uv run python examples/viewer_6dof.py --scene dominoes
    uv run python examples/viewer_6dof.py --scene domino_stress   # many cascading chains
    uv run python examples/viewer_6dof.py --scene unified         # rigid bodies + cloth + joints
    uv run python examples/viewer_6dof.py --stress      # 384-box pile
    uv run python examples/viewer_6dof.py --mega        # 1440-box pile
    # open http://localhost:8080

Knobs: substeps, position iterations, gravity, friction, restitution. The whole
scene runs on CUDA; the Performance panel shows the captured-graph step time.
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import viser

from xpbd3d import Solver6DOF

# Unit cube (±0.5) rendered per-instance at scale = 2·half_extents.
_CUBE_V = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                    [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]], np.float32)
_CUBE_F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                    [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]], np.uint32)


def _u8(c):
    return (int(np.clip(c[0], 0, 1) * 255), int(np.clip(c[1], 0, 1) * 255),
            int(np.clip(c[2], 0, 1) * 255))


# =============================================================================
# Scenes
# =============================================================================
def _solver(args):
    return Solver6DOF(dt=1 / 60, substeps=args.substeps, iterations=args.iterations,
                      gravity=(0.0, args.gravity, 0.0), floor_y=0.0,
                      friction=args.friction, restitution=args.restitution,
                      device=args.device)


def build_stack(args):
    s = _solver(args)
    h = 0.2
    rng = np.random.default_rng(0)
    for k in range(args.stack_height):
        col = (0.35 + 0.45 * rng.random(), 0.45 + 0.35 * rng.random(), 0.85)
        s.add_box((rng.uniform(-2e-3, 2e-3), h + k * (2 * h + 0.02), rng.uniform(-2e-3, 2e-3)),
                  (h, h, h), mass=1.0, color=col)
    return s


def build_dominoes(args):
    s = _solver(args)
    s.friction = max(s.friction, 0.6)
    import warp as wp
    hw, hh, hd = 0.03, 0.22, 0.12
    sp = 0.17
    n = args.n_dominoes
    for k in range(n):
        # lean the first domino forward to start the cascade
        if k == 0:
            qq = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -0.5)
            q = (qq[0], qq[1], qq[2], qq[3]); y = hh * 0.88
        else:
            q = (0.0, 0.0, 0.0, 1.0); y = hh
        col = (0.9, 0.55 - 0.4 * k / n, 0.2 + 0.6 * k / n)
        s.add_box((k * sp - n * sp / 2, y, 0.0), (hw, hh, hd), mass=1.0,
                  quaternion=q, color=col)
    return s


def build_pile(args, nx, ny, nz):
    s = _solver(args)
    h = 0.16
    rng = np.random.default_rng(0)
    sp = 2.25 * h
    for ly in range(ny):
        for ix in range(nx):
            for iz in range(nz):
                col = (0.3 + 0.5 * rng.random(), 0.4 + 0.4 * rng.random(), 0.85)
                s.add_box((ix * sp - nx * sp / 2 + rng.uniform(-1e-3, 1e-3),
                           h + ly * sp,
                           iz * sp - nz * sp / 2 + rng.uniform(-1e-3, 1e-3)),
                          (h, h, h), mass=1.0, color=col)
    return s


def build_domino_stress(args):
    """Many parallel domino chains, each kicked off at one end, all cascading at
    once — a broad-phase / contact stress test."""
    s = _solver(args)
    s.friction = max(s.friction, 0.6)
    import warp as wp
    hw, hh, hd = 0.03, 0.22, 0.10
    sp = 0.17
    nc = args.n_chains
    per = args.per_chain
    gap = 0.6                                   # spacing between chains (z)
    lean = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -0.5)
    for c in range(nc):
        z = (c - nc / 2) * gap
        base = (0.25, 0.55, 0.9) if c % 2 == 0 else (0.95, 0.6, 0.2)
        for k in range(per):
            tip = k == 0
            q = (lean[0], lean[1], lean[2], lean[3]) if tip else (0.0, 0.0, 0.0, 1.0)
            y = hh * 0.88 if tip else hh
            t = k / max(per - 1, 1)
            col = (base[0] * (0.5 + 0.5 * t), base[1], base[2] * (0.5 + 0.5 * (1 - t)))
            s.add_box((k * sp - per * sp / 2, y, z), (hw, hh, hd), mass=1.0,
                      quaternion=q, color=col)
    return s, []


def _grid_faces(R):
    """Two triangles per cell of an R×R particle grid (single winding — the mesh
    is rendered with ``side="double"``, so duplicating reversed faces would just
    create coincident coplanar geometry that z-fights and flickers)."""
    f = []
    for iy in range(R - 1):
        for ix in range(R - 1):
            a = iy * R + ix; b = iy * R + ix + 1
            c = (iy + 1) * R + ix; d = (iy + 1) * R + ix + 1
            f.append([a, c, b]); f.append([b, c, d])
    return np.asarray(f, np.uint32)


def build_unified(args):
    """One scene with all three constraint types coupled in a single substep
    loop / captured graph: rigid bodies (a block pile + a swinging bar), a
    **cloth** curtain (particles + compliant distance joints) hung from the bar,
    and an **articulated** rigid pendulum (boxes linked by rigid joints)."""
    s = _solver(args)
    s.friction = max(s.friction, 0.5)
    s.iterations = max(s.iterations, 2)         # joints converge better with ≥2
    import warp as wp
    cloth_meshes = []
    post_col = (0.42, 0.42, 0.48)

    # --- clothesline: a dynamic bar held between two static posts by joints ---
    W = 1.6
    barhalf = W / 2.0 + 0.06
    z0 = -0.7
    ypost = 1.9
    ytop = ypost + 0.5                          # post top / bar height
    post_l = s.add_box((-barhalf - 0.1, ypost, z0), (0.06, 0.5, 0.06), mass=0.0,
                       static=True, color=post_col)
    post_r = s.add_box((barhalf + 0.1, ypost, z0), (0.06, 0.5, 0.06), mass=0.0,
                       static=True, color=post_col)
    bar = s.add_box((0.0, ytop, z0), (barhalf, 0.04, 0.04), mass=2.0, color=(0.55, 0.4, 0.3))
    s.add_joint(bar, post_l, compliance=0.0, anchor_a=(-barhalf, 0.0, 0.0), anchor_b=(0.0, 0.5, 0.0))
    s.add_joint(bar, post_r, compliance=0.0, anchor_a=(barhalf, 0.0, 0.0), anchor_b=(0.0, 0.5, 0.0))

    # --- cloth curtain hung from the bar ---
    R = args.cloth_res
    sp = W / (R - 1)
    ctop = ytop - 0.05
    idx = np.empty((R, R), np.int64)
    for iy in range(R):
        for ix in range(R):
            px = -W / 2.0 + ix * sp
            py = ctop - iy * sp
            b = s.add_particle((px, py, z0), mass=0.02, radius=0.012, group=1,
                               color=(0.85, 0.22, 0.30))
            idx[iy, ix] = b.index
    for iy in range(R):
        for ix in range(R):
            if ix + 1 < R:
                s.add_joint(int(idx[iy, ix]), int(idx[iy, ix + 1]), compliance=1e-7)
            if iy + 1 < R:
                s.add_joint(int(idx[iy, ix]), int(idx[iy + 1, ix]), compliance=1e-7)
            if ix + 1 < R and iy + 1 < R:       # shear (both diagonals)
                s.add_joint(int(idx[iy, ix]), int(idx[iy + 1, ix + 1]), compliance=2e-7)
                s.add_joint(int(idx[iy + 1, ix]), int(idx[iy, ix + 1]), compliance=2e-7)
    for ix in range(R):                         # pin top row to the (movable) bar
        px = -W / 2.0 + ix * sp
        s.add_joint(int(idx[0, ix]), bar, compliance=0.0, anchor_b=(px, -0.04, 0.0))
    cloth_meshes.append({"idx": idx, "faces": _grid_faces(R), "color": (0.85, 0.22, 0.30)})

    # --- articulated rigid pendulum: boxes linked by rigid joints ---
    link = 0.30
    hx, hz = 2.4, 0.7
    hook = s.add_box((hx, ytop, hz), (0.05, 0.05, 0.05), mass=0.0, static=True, color=post_col)
    prev = hook
    for k in range(4):
        lk = s.add_box((hx + (k + 0.5) * link + link * 0.5 * k, ytop, hz),
                       (link * 0.45, 0.05, 0.05), mass=0.6, color=(0.2, 0.7 - 0.1 * k, 0.85))
        a_anchor = (0.0, 0.0, 0.0) if k == 0 else (link * 0.45, 0.0, 0.0)
        s.add_joint(prev, lk, compliance=0.0, anchor_a=a_anchor, anchor_b=(-link * 0.45, 0.0, 0.0))
        prev = lk

    # --- a small rigid block pile for the cloth/boxes to interact with ---
    rng = np.random.default_rng(0)
    hb = 0.16
    for ly in range(3):
        for ix in range(3):
            s.add_box((-2.1 + ix * 0.36 + rng.uniform(-1e-3, 1e-3), hb + ly * 0.34,
                       1.0 + rng.uniform(-1e-3, 1e-3)),
                      (hb, hb, hb), mass=1.0,
                      color=(0.3 + 0.5 * rng.random(), 0.5, 0.85))
    return s, cloth_meshes


def build_scene(args):
    """Returns ``(solver, cloth_meshes)`` — ``cloth_meshes`` is a list of
    ``{idx, faces, color}`` describing particle grids to render as sheets."""
    if args.scene == "stack":
        return build_stack(args), []
    if args.scene == "dominoes":
        return build_dominoes(args), []
    if args.scene == "domino_stress":
        return build_domino_stress(args)
    if args.scene == "unified":
        return build_unified(args)
    if args.scene == "pile":
        return build_pile(args, 4, 4, 4), []
    if args.scene == "stress":
        return build_pile(args, 8, 6, 8), []
    if args.scene == "mega":
        return build_pile(args, 12, 10, 12), []
    raise ValueError(args.scene)


# =============================================================================
# Viewer
# =============================================================================
class Viewer:
    def __init__(self, args):
        self.args = args
        self._lock = threading.RLock()      # serialise solver vs GUI-callback threads
        self.server = viser.ViserServer(host="0.0.0.0", port=args.port)
        try:
            self.server.scene.set_up_direction("+y")
        except Exception:
            pass
        self.solver, self._cloth_meshes = build_scene(args)
        self.solver._flush()

        self.server.scene.add_box("/ground", dimensions=(12.0, 0.04, 12.0),
                                  position=(0.0, -0.02, 0.0), color=(0.82, 0.82, 0.82))
        # Grid lifted a hair above the floor top (both at y=0) to kill z-fighting
        # between the coplanar grid lines and the ground surface.
        self.server.scene.add_grid("/grid", width=12.0, height=12.0, cell_size=0.5,
                                  plane="xz", position=(0.0, 0.003, 0.0))

        self._boxes_node = None
        self._build_boxes()
        self._build_cloths()
        self._build_gui()
        self._frame = 0
        self._step_ms = []

    def _build_boxes(self):
        """Instanced cubes for the rigid (non-particle) bodies only; cloth nodes
        are drawn as sheets by ``_build_cloths``."""
        if self._boxes_node is not None:
            try:
                self._boxes_node.remove()
            except Exception:
                pass
        s = self.solver
        he = np.asarray(s._he, np.float32).reshape(-1, 3)
        self._rigid_idx = np.array([b.index for b in s.bodies if not b.is_particle], np.int64)
        if len(self._rigid_idx) == 0:
            self._boxes_node = None
            return
        ri = self._rigid_idx
        scales = (2.0 * he[ri]).astype(np.float32)
        colors = np.array([_u8(s.bodies[i].color) for i in ri], np.uint8)
        pos = s.positions()[ri].astype(np.float32)
        wxyz = s.orientations()[ri][:, [3, 0, 1, 2]].astype(np.float32)
        self._boxes_node = self.server.scene.add_batched_meshes_simple(
            "/boxes", _CUBE_V, _CUBE_F, batched_positions=pos, batched_wxyzs=wxyz,
            batched_scales=scales, batched_colors=colors, flat_shading=True)

    def _build_cloths(self):
        """(Re)create each cloth sheet mesh from current particle positions.
        viser meshes have no vertex setter, so we replace the node by name."""
        P = self.solver.positions()
        for k, cm in enumerate(self._cloth_meshes):
            verts = P[cm["idx"].reshape(-1)].astype(np.float32)
            self.server.scene.add_mesh_simple(
                f"/cloth{k}", vertices=verts, faces=cm["faces"],
                color=_u8(cm["color"]), flat_shading=False, side="double")

    def _build_gui(self):
        a = self.args
        with self.server.gui.add_folder("Simulation"):
            self.g_pause = self.server.gui.add_checkbox("pause", False)
            self.g_substeps = self.server.gui.add_slider("substeps", 1, 40, 1, a.substeps,
                hint="XPBD small steps (Müller 2020): more = stiffer/stabler stacks.")
            self.g_iters = self.server.gui.add_slider("pos iterations", 1, 10, 1, a.iterations)
            self.g_gravity = self.server.gui.add_slider("gravity", -30.0, 0.0, 0.5, a.gravity)
            self.g_friction = self.server.gui.add_slider("friction μ", 0.0, 1.0, 0.02, a.friction)
            self.g_rest = self.server.gui.add_slider("restitution", 0.0, 0.9, 0.05, a.restitution)
        with self.server.gui.add_folder("Actions"):
            self.g_reset = self.server.gui.add_button("reset scene")
            self.g_drop = self.server.gui.add_button("drop a box")
            self.g_push = self.server.gui.add_button("shove (kick all)")
            self.g_reset.on_click(lambda _: self._reset())
            self.g_drop.on_click(lambda _: self._drop())
            self.g_push.on_click(lambda _: self._push())
        with self.server.gui.add_folder("Performance"):
            self.p_device = self.server.gui.add_text("device", self.solver.device)
            self.p_bodies = self.server.gui.add_text("bodies", str(self.solver.num_bodies))
            self.p_pairs = self.server.gui.add_text("contact pairs", "—")
            self.p_step = self.server.gui.add_text("step time", "—")
            self.p_cap = self.server.gui.add_text("solver capacity", "—")

        self.g_substeps.on_update(lambda _: self._set("substeps", int(self.g_substeps.value)))
        self.g_iters.on_update(lambda _: self._set("iterations", int(self.g_iters.value)))
        self.g_gravity.on_update(lambda _: self._set("gravity", (0.0, float(self.g_gravity.value), 0.0)))
        self.g_friction.on_update(lambda _: self._set("friction", float(self.g_friction.value)))
        self.g_rest.on_update(lambda _: self._set("restitution", float(self.g_rest.value)))

    def _set(self, attr, val):
        with self._lock:
            setattr(self.solver, attr, val)
            self.solver._graph = None       # scalar params are baked into the graph

    def _reset(self):
        with self._lock:
            self.solver, self._cloth_meshes = build_scene(self.args)
            self.solver.substeps = int(self.g_substeps.value)
            self.solver.iterations = int(self.g_iters.value)
            self.solver.gravity = (0.0, float(self.g_gravity.value), 0.0)
            self.solver.friction = float(self.g_friction.value)
            self.solver.restitution = float(self.g_rest.value)
            self.solver._flush()
            self._build_boxes()
            self._build_cloths()
            self._frame = 0

    def _drop(self):
        with self._lock:
            import warp as wp
            rng = np.random.default_rng()
            he = np.array([0.2, 0.14, 0.2], np.float32)
            x = float(rng.uniform(-1, 1)); z = float(rng.uniform(-1, 1))
            # Spawn clear of every existing body: sit above the tallest body whose
            # xz-footprint overlaps the new box. Rapid clicks then *stack* instead
            # of spawning two boxes interpenetrating at y=3 — an overlapping spawn
            # is resolved as a single-substep depenetration and rockets both bodies
            # off-screen at the velocity cap.
            P = self.solver.positions()
            HE = np.asarray(self.solver._he, np.float32).reshape(-1, 3)
            y = 3.0
            if len(P):
                near = ((np.abs(P[:, 0] - x) < HE[:, 0] + he[0] + 0.05) &
                        (np.abs(P[:, 2] - z) < HE[:, 2] + he[2] + 0.05))
                if near.any():
                    y = max(y, float((P[near, 1] + HE[near, 1]).max()) + he[1] + 0.15)
            ax = wp.normalize(wp.vec3(float(rng.uniform(-1, 1)), 1.0, float(rng.uniform(-1, 1))))
            qq = wp.quat_from_axis_angle(ax, float(rng.uniform(0, 1.0)))
            self.solver.add_box((x, y, z), tuple(float(v) for v in he), mass=1.5,
                                quaternion=(qq[0], qq[1], qq[2], qq[3]),
                                color=(0.2, 0.85, 0.85))
            self.solver._flush()
            self._build_boxes()

    def _push(self):
        with self._lock:
            rng = np.random.default_rng()
            for i in range(self.solver.num_bodies):
                self.solver.add_impulse(i, (float(rng.uniform(-2, 2)), float(rng.uniform(0, 1)),
                                            float(rng.uniform(-2, 2))))

    def tick(self):
        if self.g_pause.value:
            return
        t0 = time.perf_counter()
        with self._lock:
            self.solver.step()
            pos = self.solver.positions().astype(np.float32)
            wxyz = self.solver.orientations()[:, [3, 0, 1, 2]].astype(np.float32)
            node = self._boxes_node
            n_pairs = self.solver.n_pairs
        dt = time.perf_counter() - t0
        ri = self._rigid_idx
        with self.server.atomic():
            if node is not None and len(ri) == node.batched_positions.shape[0]:
                node.batched_positions = pos[ri]
                node.batched_wxyzs = wxyz[ri]
            for k, cm in enumerate(self._cloth_meshes):
                self.server.scene.add_mesh_simple(
                    f"/cloth{k}", vertices=pos[cm["idx"].reshape(-1)], faces=cm["faces"],
                    color=_u8(cm["color"]), flat_shading=False, side="double")
        self._frame += 1
        self._step_ms.append(dt * 1000.0)
        self._step_ms = self._step_ms[-30:]
        if self._frame % 5 == 0:
            ms = float(np.mean(self._step_ms))
            self.p_bodies.value = str(self.solver.num_bodies)
            self.p_pairs.value = str(n_pairs)
            self.p_step.value = f"{ms:.2f} ms"
            self.p_cap.value = f"{1000.0 / max(ms, 1e-3):.0f} Hz"

    def run(self):
        print("\nviser server running — open the URL above in a browser.\n")
        target = self.solver.dt
        try:
            while True:
                t = time.perf_counter()
                self.tick()
                spent = time.perf_counter() - t
                if spent < target:
                    time.sleep(target - spent)
        except KeyboardInterrupt:
            print("\nstopping...")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", choices=["stack", "dominoes", "domino_stress",
                                       "unified", "pile", "stress", "mega"],
                   default="pile")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--substeps", type=int, default=15)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--gravity", type=float, default=-9.81)
    p.add_argument("--friction", type=float, default=0.6)
    p.add_argument("--restitution", type=float, default=0.0)
    p.add_argument("--stack-height", type=int, default=6)
    p.add_argument("--n-dominoes", type=int, default=8)
    p.add_argument("--n-chains", type=int, default=8, help="domino_stress: parallel chains")
    p.add_argument("--per-chain", type=int, default=22, help="domino_stress: dominoes per chain")
    p.add_argument("--cloth-res", type=int, default=20, help="unified: cloth grid resolution")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--stress", action="store_true", help="384-box pile (8x6x8)")
    p.add_argument("--mega", action="store_true", help="1440-box pile (12x10x12)")
    args = p.parse_args()
    if args.stress:
        args.scene = "stress"
    if args.mega:
        args.scene = "mega"
    Viewer(args).run()


if __name__ == "__main__":
    main()
