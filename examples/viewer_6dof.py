"""Interactive 3D viewer for the 6-DOF rigid-body XPBD solver (viser).

The 6-DOF counterpart of ``viewer.py``: real rigid **boxes** with quaternion
orientation, rotated inertia and OBB contacts (Müller et al. 2020), running on
the GPU with a CUDA-graph-captured substep loop. Mirrors the AVBD ``Solver6DOF``
viewer — stacks, dominoes, and big block piles you can watch settle.

    uv run python examples/viewer_6dof.py --scene stack
    uv run python examples/viewer_6dof.py --scene dominoes
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


def build_scene(args):
    if args.scene == "stack":
        return build_stack(args)
    if args.scene == "dominoes":
        return build_dominoes(args)
    if args.scene == "pile":
        return build_pile(args, 4, 4, 4)
    if args.scene == "stress":
        return build_pile(args, 8, 6, 8)
    if args.scene == "mega":
        return build_pile(args, 12, 10, 12)
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
        self.solver = build_scene(args)
        self.solver._flush()

        self.server.scene.add_box("/ground", dimensions=(12.0, 0.04, 12.0),
                                  position=(0.0, -0.02, 0.0), color=(0.82, 0.82, 0.82))
        # Grid lifted a hair above the floor top (both at y=0) to kill z-fighting
        # between the coplanar grid lines and the ground surface.
        self.server.scene.add_grid("/grid", width=12.0, height=12.0, cell_size=0.5,
                                  plane="xz", position=(0.0, 0.003, 0.0))

        self._boxes_node = None
        self._build_boxes()
        self._build_gui()
        self._frame = 0
        self._step_ms = []

    def _build_boxes(self):
        if self._boxes_node is not None:
            try:
                self._boxes_node.remove()
            except Exception:
                pass
        s = self.solver
        n = s.num_bodies
        he = np.asarray(s._he, np.float32).reshape(-1, 3)
        scales = (2.0 * he).astype(np.float32)
        colors = np.array([_u8(b.color) for b in s.bodies], np.uint8)
        pos = s.positions().astype(np.float32)
        wxyz = s.orientations()[:, [3, 0, 1, 2]].astype(np.float32)
        self._boxes_node = self.server.scene.add_batched_meshes_simple(
            "/boxes", _CUBE_V, _CUBE_F, batched_positions=pos, batched_wxyzs=wxyz,
            batched_scales=scales, batched_colors=colors, flat_shading=True)

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
            self.p_bodies = self.server.gui.add_text("boxes", str(self.solver.num_bodies))
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
            self.solver = build_scene(self.args)
            self.solver.substeps = int(self.g_substeps.value)
            self.solver.iterations = int(self.g_iters.value)
            self.solver.gravity = (0.0, float(self.g_gravity.value), 0.0)
            self.solver.friction = float(self.g_friction.value)
            self.solver.restitution = float(self.g_rest.value)
            self.solver._flush()
            self._build_boxes()
            self._frame = 0

    def _drop(self):
        with self._lock:
            import warp as wp
            rng = np.random.default_rng()
            ax = wp.normalize(wp.vec3(float(rng.uniform(-1, 1)), 1.0, float(rng.uniform(-1, 1))))
            qq = wp.quat_from_axis_angle(ax, float(rng.uniform(0, 1.0)))
            self.solver.add_box((float(rng.uniform(-1, 1)), 3.0, float(rng.uniform(-1, 1))),
                                (0.2, 0.14, 0.2), mass=1.5,
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
        if node is not None and len(pos) == node.batched_positions.shape[0]:
            with self.server.atomic():
                node.batched_positions = pos
                node.batched_wxyzs = wxyz
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
    p.add_argument("--scene", choices=["stack", "dominoes", "pile", "stress", "mega"],
                   default="pile")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--substeps", type=int, default=15)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--gravity", type=float, default=-9.81)
    p.add_argument("--friction", type=float, default=0.6)
    p.add_argument("--restitution", type=float, default=0.0)
    p.add_argument("--stack-height", type=int, default=6)
    p.add_argument("--n-dominoes", type=int, default=8)
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
