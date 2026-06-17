"""XPBD physics on a parsed robot (URDF/MJCF) — viser.

Loads a Unitree robot with :mod:`xpbd3d.robot`, maps it onto the 6-DOF XPBD
``Solver6DOF`` (``robot/xpbd_build.py``) and simulates it live. Each rigid
segment is mapped to the solver's native collision shapes taken directly from the
file's collision geometry — ``<box>``→box, ``<cylinder>``→cylinder (flat caps),
``<sphere>``→sphere, ``<capsule>``→capsule. Revolute joints become two
coincident-anchor distance constraints along the hinge axis (1 free DOF). Toggle
"show physics shapes" to see the exact collision proxies the solver simulates.

    uv run python examples/sim_robot_xpbd.py --format mjcf --scene hang
    uv run python examples/sim_robot_xpbd.py --format urdf --scene drop
    # open http://localhost:8080

* hang — base pinned in the air; the legs swing down under gravity and **settle**
  into a hanging rest pose (a pinned articulated chain is a pendulum, so a little
  velocity damping is applied — without it the undamped legs swing forever).
* drop — base free; the whole robot falls and its shape proxies land on the floor.

Note on self-intersection: all robot links share one collision ``group`` so the
solver runs **no link-vs-link self-collision** (a single-int group can't exclude
only joint-adjacent pairs in a serial chain). Limbs can therefore pass through
each other; only link-vs-floor contact is resolved. See ``diag_robot_hang.py``.
"""

from __future__ import annotations

import argparse
import os
import threading
import time

import numpy as np
import trimesh
import viser

from xpbd3d.robot import load_mjcf, load_urdf
from xpbd3d.robot.viser_render import RobotView
from xpbd3d.robot.xpbd_build import (build_xpbd, link_world_transforms,
                                     proxy_world_poses, read_state)


def _proxy_mesh(shape_type, he, radius, half_len):
    """Unit-placed trimesh for a collision proxy (centred, axis along +Z)."""
    if shape_type == 2:                              # cylinder (flat caps)
        return trimesh.creation.cylinder(radius=radius, height=2.0 * half_len, sections=20)
    if shape_type == 1:
        if half_len < 1e-6:
            return trimesh.creation.icosphere(subdivisions=2, radius=radius)
        m = trimesh.creation.capsule(height=2.0 * half_len, radius=radius, count=[12, 12])
        m.apply_translation((0.0, 0.0, -half_len))   # trimesh capsule starts at z=0
        return m
    return trimesh.creation.box(extents=(2.0 * he[0], 2.0 * he[1], 2.0 * he[2]))

MJCF_ROOT = "/home/nan/Desktop/unitree_robots"
URDF_ROOT = "/home/nan/Desktop/urdf"


def resolve_path(fmt: str, robot: str) -> str:
    if fmt == "mjcf":
        return os.path.join(MJCF_ROOT, robot, f"{robot}.xml")
    return os.path.join(URDF_ROOT, f"{robot}_description", "urdf", f"{robot}_description.urdf")


def home_pose(model) -> dict:
    """Natural standing crouch (hip 0, thigh ~0.9, calf ~-1.8), clamped to limits."""
    q = {}
    for j in model.actuated_joints:
        n = j.name.lower()
        v = 0.9 if "thigh" in n else (-1.8 if ("calf" in n or "knee" in n) else 0.0)
        if j.lower < j.upper:
            v = float(np.clip(v, j.lower, j.upper))
        q[j.name] = v
    return q


class RobotSim:
    def __init__(self, args):
        self.args = args
        self._lock = threading.RLock()
        path = args.path or resolve_path(args.format, args.robot)
        print(f"[robot] loading {args.format.upper()}: {path}")
        self.model = (load_mjcf if args.format == "mjcf" else load_urdf)(path)
        print(f"[robot] {self.model}")

        self.server = viser.ViserServer(host="0.0.0.0", port=args.port)
        try:
            self.server.scene.set_up_direction("+z")
        except Exception:
            pass
        self.server.scene.add_grid("/grid", width=4.0, height=4.0, cell_size=0.25,
                                   plane="xy", position=(0.0, 0.0, 0.0))

        self.view = RobotView(self.server, self.model, lift=False,
                              show_visual=True, show_collision=False)
        self._proxy_nodes = []
        self._build_physics(args.scene)
        self._build_gui()

    # -- physics --------------------------------------------------------------
    def _scene_damping(self, scene: str):
        """Velocity damping (per substep). A base-pinned articulated chain is an
        undamped pendulum, so the legs need a little angular damping to settle
        into a hanging rest pose instead of swinging forever. The free-falling
        ``drop`` keeps linear damping at 0 so the fall stays natural. CLI flags
        override."""
        lin = 0.01 if scene == "hang" else 0.0
        ang = 0.03 if scene == "hang" else 0.02
        if self.args.lin_damp is not None:
            lin = self.args.lin_damp
        if self.args.ang_damp is not None:
            ang = self.args.ang_damp
        return lin, ang

    def _build_physics(self, scene: str):
        self.scene = scene
        q0 = home_pose(self.model)
        lin, ang = self._scene_damping(scene)
        self.phys = build_xpbd(
            self.model, q0=q0, base_static=(scene == "hang"),
            device=self.args.device, substeps=self.args.substeps,
            iterations=self.args.iterations, friction=self.args.friction,
            lin_damp=lin, ang_damp=ang,
            start_clearance=0.12 if scene == "drop" else 0.06)
        self.solver = self.phys.solver
        self.solver._flush()
        self.view.update_world(link_world_transforms(self.phys))
        self._build_proxies()

    def _build_proxies(self):
        for h in self._proxy_nodes:
            try:
                h.remove()
            except Exception:
                pass
        self._proxy_nodes = []
        poses = proxy_world_poses(self.phys)
        for k, (bidx, st, he, r, hl) in enumerate(self.phys.proxies):
            m = _proxy_mesh(st, he, r, hl)
            center, wxyz = poses[k]
            h = self.server.scene.add_mesh_simple(
                f"/phys/{k}", np.asarray(m.vertices, np.float32),
                np.asarray(m.faces, np.uint32), color=(60, 200, 255),
                wireframe=True, opacity=0.7, position=center, wxyz=wxyz, visible=False)
            self._proxy_nodes.append(h)

    # -- GUI ------------------------------------------------------------------
    def _build_gui(self):
        a = self.args
        with self.server.gui.add_folder("Simulation"):
            self.g_pause = self.server.gui.add_checkbox("pause", False)
            self.g_scene = self.server.gui.add_dropdown("scene", ("hang", "drop"),
                                                        initial_value=self.scene)
            self.g_reset = self.server.gui.add_button("reset")
            self.g_substeps = self.server.gui.add_slider("substeps", 5, 40, 1, a.substeps)
            self.g_iters = self.server.gui.add_slider("iterations", 1, 16, 1, a.iterations)
            self.g_gravity = self.server.gui.add_slider("gravity", -20.0, 0.0, 0.5, -9.81)
            self.g_friction = self.server.gui.add_slider("friction μ", 0.0, 1.5, 0.05, a.friction)
            lin0, ang0 = self._scene_damping(self.scene)
            self.g_lindamp = self.server.gui.add_slider("linear damping", 0.0, 0.1, 0.005, lin0)
            self.g_angdamp = self.server.gui.add_slider("angular damping", 0.0, 0.1, 0.005, ang0)
        with self.server.gui.add_folder("Display"):
            self.g_visual = self.server.gui.add_checkbox("show visual", True)
            self.g_collision = self.server.gui.add_checkbox("show collision", False)
            self.g_boxes = self.server.gui.add_checkbox("show physics shapes", False)
        with self.server.gui.add_folder("Info"):
            self.p_device = self.server.gui.add_text("device", str(self.solver.device))
            self.p_bodies = self.server.gui.add_text("bodies (clusters)", str(self.solver.num_bodies))
            self.p_joints = self.server.gui.add_text("joint constraints", str(self.solver.num_joints))
            self.p_step = self.server.gui.add_text("step time", "—")

        self.g_scene.on_update(lambda _: self._reset(self.g_scene.value))
        self.g_reset.on_click(lambda _: self._reset(self.g_scene.value))
        self.g_substeps.on_update(lambda _: self._set("substeps", int(self.g_substeps.value)))
        self.g_iters.on_update(lambda _: self._set("iterations", int(self.g_iters.value)))
        self.g_gravity.on_update(lambda _: self._set("gravity", (0.0, float(self.g_gravity.value), 0.0)))
        self.g_friction.on_update(lambda _: self._set("friction", float(self.g_friction.value)))
        self.g_lindamp.on_update(lambda _: self._set("lin_damp", float(self.g_lindamp.value)))
        self.g_angdamp.on_update(lambda _: self._set("ang_damp", float(self.g_angdamp.value)))
        self.g_visual.on_update(lambda _: self.view.set_visual_visible(self.g_visual.value))
        self.g_collision.on_update(lambda _: self.view.set_collision_visible(self.g_collision.value))
        self.g_boxes.on_update(lambda _: [setattr(h, "visible", self.g_boxes.value) for h in self._proxy_nodes])

    def _set(self, attr, val):
        with self._lock:
            setattr(self.solver, attr, val)
            self.solver._graph = None       # scalar params are baked into the graph

    def _reset(self, scene):
        with self._lock:
            self._build_physics(scene)
            lin0, ang0 = self._scene_damping(scene)
            self.g_lindamp.value = lin0
            self.g_angdamp.value = ang0
            self.solver.substeps = int(self.g_substeps.value)
            self.solver.iterations = int(self.g_iters.value)
            self.solver.gravity = (0.0, float(self.g_gravity.value), 0.0)
            self.solver.friction = float(self.g_friction.value)
            self.solver.lin_damp = lin0
            self.solver.ang_damp = ang0
            self.p_bodies.value = str(self.solver.num_bodies)
            self.p_joints.value = str(self.solver.num_joints)
            for h in self._proxy_nodes:
                h.visible = self.g_boxes.value

    # -- loop -----------------------------------------------------------------
    def tick(self):
        if self.g_pause.value:
            return
        t0 = time.perf_counter()
        with self._lock:
            self.solver.step()
            state = read_state(self.phys)          # single host readback per frame
            world = link_world_transforms(self.phys, state)
            poses = proxy_world_poses(self.phys, state) if self.g_boxes.value else None
        dt = time.perf_counter() - t0
        self.view.update_world(world)
        if poses is not None:
            with self.server.atomic():
                for h, (center, wxyz) in zip(self._proxy_nodes, poses):
                    h.position = center
                    h.wxyz = wxyz
        self.p_step.value = f"{dt * 1000.0:.2f} ms"

    def run(self):
        print("\nviser server running — open the URL above in a browser.\n")
        target = 1.0 / 60.0
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
    p = argparse.ArgumentParser(description="XPBD physics robot viewer (viser)")
    p.add_argument("--format", choices=["mjcf", "urdf"], default="mjcf")
    p.add_argument("--robot", default="go2")
    p.add_argument("--path", default=None)
    p.add_argument("--scene", choices=["hang", "drop"], default="hang")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--substeps", type=int, default=20)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--friction", type=float, default=0.8)
    p.add_argument("--lin-damp", type=float, default=None,
                   help="linear velocity damping per substep (default: scene-dependent)")
    p.add_argument("--ang-damp", type=float, default=None,
                   help="angular velocity damping per substep (default: scene-dependent)")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    RobotSim(args).run()


if __name__ == "__main__":
    main()
