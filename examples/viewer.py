"""Interactive 3D XPBD viewer (viser, browser-based).

Mirrors the AVBD demo: a viser server running the Warp XPBD solver on the GPU
(or CPU), with live "knobs" in a GUI panel. Three scenes:

    --scene chain   N-link chain hung from a static anchor + droppable spheres
    --scene cloth   a draped cloth sheet (the XPBD showcase) pinned at corners
    --scene stack   a pile of spheres on a floor (the Jacobi-on-GPU benchmark)

Run:

    uv run python examples/viewer.py --scene cloth --device cuda:0
    # open http://localhost:8080 (URL also printed)

The headline XPBD knob is **stiffness / compliance**: drag it and the same
scene goes from infinitely stiff to soft — at a fixed iteration count, which is
exactly what XPBD buys you over classic PBD (paper Fig. 2). **substeps** is the
"small steps" stability knob (Müller 2020); **solve mode** flips between colored
Gauss-Seidel and the paper's fully-parallel Jacobi (default on CUDA).
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import trimesh
import viser

from xpbd3d import Body, Shape, Solver

# A unit icosphere reused as the instanced-sphere render primitive.
_ICO = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
_ICO_V = np.asarray(_ICO.vertices, dtype=np.float32)
_ICO_F = np.asarray(_ICO.faces, dtype=np.uint32)


# =============================================================================
# Scene builders — each returns (solver, spec) where spec describes rendering.
# =============================================================================
def _u8(c):
    return (int(np.clip(c[0], 0, 1) * 255),
            int(np.clip(c[1], 0, 1) * 255),
            int(np.clip(c[2], 0, 1) * 255))


def build_chain(args):
    s = Solver(dt=1 / 60, substeps=args.substeps, iterations=args.iterations,
               gravity=(0.0, args.gravity, 0.0), device=args.device,
               solve_mode=args.solve_mode)
    link = args.link
    top = (0.0, args.top_y, 0.0)
    spheres = []   # (body_idx, radius, color_u8)
    links = []     # (body_idx_a, body_idx_b)

    anchor = s.add_particle(top, mass=0.0, collide=False,
                            shape=Shape("sphere", (0.12,)))
    spheres.append((anchor.index, 0.12, _u8((0.9, 0.2, 0.2))))
    prev = anchor
    for i in range(1, args.n + 1):
        pos = (top[0], top[1] - i * link, top[2])
        b = s.add_particle(pos, mass=1.0, shape=Shape("sphere", (0.1,)))
        s.add_floor_contact(b, floor_y=0.1, friction=args.friction)
        s.add_distance(prev, b, rest=link, compliance=args.compliance)
        spheres.append((b.index, 0.1, _u8((0.3, 0.5, 0.9))))
        links.append((prev.index, b.index))
        prev = b
    # heavy bob
    bob = s.add_particle((top[0], top[1] - (args.n + 1) * link, top[2]),
                         mass=args.heavy_mass, shape=Shape("sphere", (0.18,)))
    s.add_floor_contact(bob, floor_y=0.18, friction=args.friction)
    s.add_distance(prev, bob, rest=link, compliance=args.compliance)
    spheres.append((bob.index, 0.18, _u8((0.7, 0.4, 0.1))))
    links.append((prev.index, bob.index))

    s.enable_self_collision(True, default_friction=args.friction)
    spec = {"spheres": spheres, "links": links, "cloth": None,
            "compliance_targets": True, "droppable": True}
    return s, spec


def build_cloth(args):
    s = Solver(dt=1 / 60, substeps=args.substeps, iterations=args.iterations,
               gravity=(0.0, args.gravity, 0.0), device=args.device,
               solve_mode=args.solve_mode, damping=0.01)
    R = args.cloth_res
    h = args.cloth_size / (R - 1)
    y0 = args.top_y
    grid = np.empty((R, R), dtype=np.int64)
    # Lay the sheet in the XZ plane (a flat hanging curtain), tilted slightly so
    # it sags visibly under gravity. Pin the two top corners (static).
    for iy in range(R):
        for ix in range(R):
            px = (ix - (R - 1) / 2) * h
            pz = (iy - (R - 1) / 2) * h
            pinned = (iy == 0 and ix in (0, R - 1))
            mass = 0.0 if pinned else args.cloth_mass
            b = s.add_particle((px, y0, pz), mass=mass, collide=False,
                               shape=Shape("sphere", (h * 0.3,)))
            grid[iy, ix] = b.index

    def link(a, b, comp):
        s.add_distance(Body(int(a)), Body(int(b)), compliance=comp)

    comp = args.compliance
    for iy in range(R):
        for ix in range(R):
            if ix + 1 < R:
                link(grid[iy, ix], grid[iy, ix + 1], comp)              # structural
            if iy + 1 < R:
                link(grid[iy, ix], grid[iy + 1, ix], comp)              # structural
            if ix + 1 < R and iy + 1 < R:
                link(grid[iy, ix], grid[iy + 1, ix + 1], comp * 2.0)    # shear
                link(grid[iy, ix + 1], grid[iy + 1, ix], comp * 2.0)    # shear
            if ix + 2 < R:
                link(grid[iy, ix], grid[iy, ix + 2], comp * 4.0)        # bend
            if iy + 2 < R:
                link(grid[iy, ix], grid[iy + 2, ix], comp * 4.0)        # bend

    # Triangulate the grid for the rendered mesh.
    faces = []
    for iy in range(R - 1):
        for ix in range(R - 1):
            a = grid[iy, ix]; b = grid[iy, ix + 1]
            c = grid[iy + 1, ix]; d = grid[iy + 1, ix + 1]
            faces.append((a, c, b)); faces.append((b, c, d))
    faces = np.asarray(faces, dtype=np.uint32)
    pinned_idx = [int(grid[0, 0]), int(grid[0, R - 1])]
    spec = {"spheres": [(p, h * 0.35, _u8((0.9, 0.2, 0.2))) for p in pinned_idx],
            "links": None, "cloth": (grid.reshape(-1), faces),
            "compliance_targets": True, "droppable": False,
            "cloth_grid": grid}
    return s, spec


def build_stack(args):
    s = Solver(dt=1 / 60, substeps=args.substeps, iterations=args.iterations,
               gravity=(0.0, args.gravity, 0.0), device=args.device,
               solve_mode=args.solve_mode)
    spheres = []
    rng = np.random.default_rng(0)
    r = 0.16
    nx = args.stack_nx
    rows = args.stack_rows
    for ly in range(rows):
        for ix in range(nx):
            for iz in range(nx):
                px = (ix - (nx - 1) / 2) * (2.3 * r) + float(rng.uniform(-2e-3, 2e-3))
                pz = (iz - (nx - 1) / 2) * (2.3 * r) + float(rng.uniform(-2e-3, 2e-3))
                py = r + ly * (2.3 * r)
                col = (0.3 + 0.5 * rng.random(), 0.4 + 0.4 * rng.random(), 0.85)
                b = s.add_particle((px, py, pz), mass=1.0,
                                   shape=Shape("sphere", (r,)), friction=args.friction)
                s.add_floor_contact(b, floor_y=r, friction=args.friction)
                spheres.append((b.index, r, _u8(col)))
    s.enable_self_collision(True, default_friction=args.friction)
    spec = {"spheres": spheres, "links": None, "cloth": None,
            "compliance_targets": False, "droppable": True}
    return s, spec


SCENES = {"chain": build_chain, "cloth": build_cloth, "stack": build_stack}


# =============================================================================
# Viewer
# =============================================================================
class Viewer:
    def __init__(self, args):
        self.args = args
        self.server = viser.ViserServer(host="0.0.0.0", port=args.port)
        try:
            self.server.scene.set_up_direction("+y")
        except Exception:
            pass
        self.solver, self.spec = SCENES[args.scene](args)
        self.solver._flush()

        # ground + grid
        self.server.scene.add_box("/ground", dimensions=(10.0, 0.04, 10.0),
                                  position=(0.0, -0.02, 0.0), color=(0.82, 0.82, 0.82))
        self.server.scene.add_grid("/grid", width=10.0, height=10.0,
                                  cell_size=0.5, plane="xz")

        self._sphere_node = None
        self._sphere_idx = np.zeros(0, np.int64)
        self._link_node = None
        self._cloth_node = None
        self._build_render()
        self._build_gui()

        self._frame = 0
        self._step_ms = []
        self._wall_ms = []
        self._last_tick = time.perf_counter()

    # ---- render layer -------------------------------------------------------
    def _build_render(self):
        sp = self.spec
        pos = self.solver.positions()
        if sp["spheres"]:
            idx = np.array([b for b, _, _ in sp["spheres"]], dtype=np.int64)
            radii = np.array([[r, r, r] for _, r, _ in sp["spheres"]], np.float32)
            cols = np.array([c for _, _, c in sp["spheres"]], np.uint8)
            wxyz = np.tile(np.array([1, 0, 0, 0], np.float32), (len(idx), 1))
            self._sphere_node = self.server.scene.add_batched_meshes_simple(
                "/spheres", _ICO_V, _ICO_F,
                batched_positions=pos[idx].astype(np.float32),
                batched_wxyzs=wxyz, batched_scales=radii, batched_colors=cols,
                flat_shading=True)
            self._sphere_idx = idx
        if sp["links"]:
            self._link_pairs = np.array(sp["links"], dtype=np.int64)
            self._link_node = self.server.scene.add_line_segments(
                "/links", points=self._link_points(),
                colors=np.array([60, 90, 200], np.uint8), line_width=3.0)
        if sp["cloth"]:
            self._cloth_idx, self._cloth_faces = sp["cloth"]
            self._emit_cloth()

    def _link_points(self):
        pos = self.solver.positions()
        out = np.zeros((len(self._link_pairs), 2, 3), np.float32)
        out[:, 0] = pos[self._link_pairs[:, 0]]
        out[:, 1] = pos[self._link_pairs[:, 1]]
        return out

    def _emit_cloth(self):
        pos = self.solver.positions()
        self._cloth_node = self.server.scene.add_mesh_simple(
            "/cloth", vertices=pos[self._cloth_idx].astype(np.float32),
            faces=self._cloth_faces, color=(80, 140, 230),
            side="double", flat_shading=False, material="standard")

    # ---- GUI ----------------------------------------------------------------
    def _build_gui(self):
        a = self.args
        with self.server.gui.add_folder("Simulation"):
            self.g_pause = self.server.gui.add_checkbox("pause", False)
            self.g_substeps = self.server.gui.add_slider(
                "substeps", 1, 40, 1, a.substeps,
                hint="XPBD 'small steps' (Müller 2020): more substeps = stiffer "
                     "& more stable at fixed total work.")
            self.g_iters = self.server.gui.add_slider(
                "iterations / substep", 1, 20, 1, a.iterations)
            self.g_gravity = self.server.gui.add_slider("gravity", -30.0, 0.0, 0.5, a.gravity)
            self.g_friction = self.server.gui.add_slider("friction μ", 0.0, 1.0, 0.02, a.friction)
            self.g_mode = self.server.gui.add_dropdown(
                "solve mode", options=("jacobi", "gs"), initial_value=a.solve_mode,
                hint="jacobi = paper's fully-parallel GPU mode (no coloring); "
                     "gs = colored Gauss-Seidel (better convergence/iter).")
        if self.spec["compliance_targets"]:
            with self.server.gui.add_folder("Stiffness (the XPBD knob)"):
                self.g_logc = self.server.gui.add_slider(
                    "log10 compliance", -8.0, -1.0, 0.1, math.log10(max(a.compliance, 1e-8)),
                    hint="Compliance α = 1/stiffness (m/N). XPBD makes the "
                         "resulting stiffness independent of iteration count "
                         "(paper Fig. 2). Left = rigid, right = soft.")
                self.g_logc.on_update(self._compliance_changed)
        else:
            self.g_logc = None
        with self.server.gui.add_folder("Actions"):
            self.g_reset = self.server.gui.add_button("reset scene")
            self.g_kick = self.server.gui.add_button("kick / wind")
            if self.spec["droppable"]:
                self.g_drop = self.server.gui.add_button("drop a sphere")
                self.g_drop.on_click(lambda _: self._drop())
            self.g_reset.on_click(lambda _: self._reset())
            self.g_kick.on_click(lambda _: self._kick())
        with self.server.gui.add_folder("Performance"):
            self.p_device = self.server.gui.add_text("device", self.solver.device)
            self.p_bodies = self.server.gui.add_text("bodies", str(self.solver.num_bodies))
            self.p_cons = self.server.gui.add_text("constraints", "—")
            self.p_colors = self.server.gui.add_text("solve", "—")
            self.p_step = self.server.gui.add_text("step time", "—")
            self.p_cap = self.server.gui.add_text("solver capacity", "—")
            self.p_wall = self.server.gui.add_text("wall tick", "—")

        self.g_substeps.on_update(lambda _: setattr(self.solver, "substeps", int(self.g_substeps.value)))
        self.g_iters.on_update(lambda _: setattr(self.solver, "iterations", int(self.g_iters.value)))
        self.g_gravity.on_update(lambda _: setattr(self.solver, "gravity", (0.0, float(self.g_gravity.value), 0.0)))
        self.g_friction.on_update(lambda _: self.solver.set_all_friction(float(self.g_friction.value)))
        self.g_mode.on_update(self._mode_changed)

    def _compliance_changed(self, _):
        self.solver.set_all_compliance(10.0 ** float(self.g_logc.value))

    def _mode_changed(self, _):
        # Switching mode needs a recolor for gs; force a flush.
        self.solver.solve_mode = str(self.g_mode.value)
        self.solver._dirty = True
        self.solver._flush()

    def _kick(self):
        if self.spec["cloth"]:
            # wind: a sideways impulse on every free cloth particle
            for i in range(self.solver.num_bodies):
                self.solver.add_impulse(Body(i), (2.5, 0.0, 1.5))
        else:
            rng = np.random.default_rng()
            for i in range(self.solver.num_bodies):
                self.solver.add_impulse(Body(i), (float(rng.uniform(-3, 3)),
                                                   float(rng.uniform(0, 2)),
                                                   float(rng.uniform(-3, 3))))

    def _drop(self):
        rng = np.random.default_rng()
        r = 0.18
        pos = (float(rng.uniform(-1, 1)), self.args.top_y + 1.0, float(rng.uniform(-1, 1)))
        b = self.solver.add_particle(pos, mass=1.5, shape=Shape("sphere", (r,)),
                                     friction=float(self.g_friction.value))
        self.solver.add_floor_contact(b, floor_y=r, friction=float(self.g_friction.value))
        self.spec["spheres"].append((b.index, r, _u8((0.2, 0.85, 0.85))))
        self.solver._flush()
        self._rebuild_spheres()

    def _rebuild_spheres(self):
        if self._sphere_node is not None:
            try:
                self._sphere_node.remove()
            except Exception:
                pass
        sp = self.spec["spheres"]
        pos = self.solver.positions()
        idx = np.array([b for b, _, _ in sp], dtype=np.int64)
        radii = np.array([[r, r, r] for _, r, _ in sp], np.float32)
        cols = np.array([c for _, _, c in sp], np.uint8)
        wxyz = np.tile(np.array([1, 0, 0, 0], np.float32), (len(idx), 1))
        self._sphere_node = self.server.scene.add_batched_meshes_simple(
            "/spheres", _ICO_V, _ICO_F, batched_positions=pos[idx].astype(np.float32),
            batched_wxyzs=wxyz, batched_scales=radii, batched_colors=cols, flat_shading=True)
        self._sphere_idx = idx

    def _reset(self):
        for n in (self._sphere_node, self._link_node, self._cloth_node):
            if n is not None:
                try:
                    n.remove()
                except Exception:
                    pass
        self._sphere_node = self._link_node = self._cloth_node = None
        self.solver, self.spec = SCENES[self.args.scene](self.args)
        self.solver.substeps = int(self.g_substeps.value)
        self.solver.iterations = int(self.g_iters.value)
        self.solver.gravity = (0.0, float(self.g_gravity.value), 0.0)
        self.solver.solve_mode = str(self.g_mode.value)
        self.solver._flush()
        if self.g_logc is not None:
            self.solver.set_all_compliance(10.0 ** float(self.g_logc.value))
        self._build_render()
        self._frame = 0

    # ---- tick ---------------------------------------------------------------
    def tick(self):
        if self.g_pause.value:
            return
        t0 = time.perf_counter()
        self.solver.step()
        step_dt = time.perf_counter() - t0

        pos = self.solver.positions()
        with self.server.atomic():
            if self._sphere_node is not None and len(self._sphere_idx):
                self._sphere_node.batched_positions = pos[self._sphere_idx].astype(np.float32)
            if self._link_node is not None:
                self._link_node.points = self._link_points()
            if self._cloth_node is not None:
                self._emit_cloth()

        self._frame += 1
        self._step_ms.append(step_dt * 1000.0)
        self._step_ms = self._step_ms[-30:]
        now = time.perf_counter()
        self._wall_ms.append((now - self._last_tick) * 1000.0)
        self._wall_ms = self._wall_ms[-30:]
        self._last_tick = now
        if self._frame % 5 == 0:
            step = float(np.mean(self._step_ms))
            wall = float(np.mean(self._wall_ms))
            self.p_bodies.value = str(self.solver.num_bodies)
            self.p_cons.value = str(self.solver.num_constraints)
            if self.solver.solve_mode == "gs":
                self.p_colors.value = f"gs · {self.solver.num_colors} colors"
            else:
                self.p_colors.value = "jacobi (no coloring)"
            self.p_step.value = f"{step:.2f} ms"
            self.p_cap.value = f"{1000.0 / max(step, 1e-3):.0f} Hz"
            self.p_wall.value = f"{1000.0 / max(wall, 1e-3):.0f} Hz"

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
    p.add_argument("--scene", choices=list(SCENES), default="cloth")
    p.add_argument("--device", default="cuda:0",
                   help="Warp device: 'cuda:0' (default) or 'cpu'.")
    p.add_argument("--solve-mode", choices=("auto", "gs", "jacobi"), default="auto")
    p.add_argument("--substeps", type=int, default=15)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--gravity", type=float, default=-9.81)
    p.add_argument("--friction", type=float, default=0.4)
    p.add_argument("--compliance", type=float, default=1e-6)
    p.add_argument("--port", type=int, default=8080)
    # chain
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--link", type=float, default=0.35)
    p.add_argument("--heavy-mass", type=float, default=5.0)
    p.add_argument("--top-y", type=float, default=3.2)
    # cloth
    p.add_argument("--cloth-res", type=int, default=24)
    p.add_argument("--cloth-size", type=float, default=2.4)
    p.add_argument("--cloth-mass", type=float, default=0.02)
    # stack
    p.add_argument("--stack-nx", type=int, default=5)
    p.add_argument("--stack-rows", type=int, default=4)
    # AVBD-parity stress presets (mirror examples/viewer.py --stress in ../AVBD):
    #   --stress : 8×8×6 = 384-body pile  (AVBD's --stress is 8×8×6 ≈ 414)
    #   --mega   : 16×16×8 = 2048-body pile (AVBD's largest reported is 16×16×8)
    p.add_argument("--stress", action="store_true",
                   help="AVBD-parity 384-body sphere pile (scene=stack, 8x8x6)")
    p.add_argument("--mega", action="store_true",
                   help="even bigger 2048-body pile (scene=stack, 16x16x8)")
    args = p.parse_args()
    if args.stress or args.mega:
        args.scene = "stack"
        args.iterations = max(args.iterations, 2)
        args.substeps = 12
        if args.mega:
            args.stack_nx, args.stack_rows = 16, 8
        else:
            args.stack_nx, args.stack_rows = 8, 6
    if args.solve_mode == "auto":
        args.solve_mode = "jacobi" if str(args.device).startswith("cuda") else "gs"
    Viewer(args).run()


if __name__ == "__main__":
    main()
