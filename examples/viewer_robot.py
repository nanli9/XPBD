"""Interactive robot viewer — load a URDF or MJCF model and show it in viser.

Parses a Unitree robot description with the project's own loaders
(:mod:`xpbd3d.robot`) into a common ``RobotModel``, then renders it Z-up with
forward-kinematics joint sliders and visual/collision visibility toggles. This
is the *visualisation* milestone — no solver is involved yet; XPBD/AVBD physics
is a separate follow-up.

    uv run python examples/viewer_robot.py --format mjcf --robot go2
    uv run python examples/viewer_robot.py --format urdf --robot go2
    # then open http://localhost:8080

Drag the per-joint sliders to pose the legs, toggle "show collision" to overlay
the collision primitives, or enable "animate" for a hands-free trot wiggle.
"""

from __future__ import annotations

import argparse
import math
import os
import threading
import time

import numpy as np
import viser

from xpbd3d.robot import load_mjcf, load_urdf
from xpbd3d.robot.viser_render import RobotView

# Where the robot descriptions live (siblings of the XPBD repo).
MJCF_ROOT = "/home/nan/Desktop/unitree_robots"
URDF_ROOT = "/home/nan/Desktop/urdf"


def resolve_path(fmt: str, robot: str) -> str:
    if fmt == "mjcf":
        return os.path.join(MJCF_ROOT, robot, f"{robot}.xml")
    return os.path.join(URDF_ROOT, f"{robot}_description", "urdf", f"{robot}_description.urdf")


def home_pose(model) -> dict:
    """A natural standing crouch for quadrupeds (hip 0, thigh ~0.9, calf ~-1.8),
    clamped to each joint's limits. Falls back to 0 for unrecognised joints."""
    q = {}
    for j in model.actuated_joints:
        n = j.name.lower()
        if "thigh" in n:
            v = 0.9
        elif "calf" in n or "knee" in n:
            v = -1.8
        else:
            v = 0.0
        if j.lower < j.upper:
            v = float(np.clip(v, j.lower, j.upper))
        q[j.name] = v
    return q


class RobotViewer:
    def __init__(self, args):
        self.args = args
        self._lock = threading.RLock()
        path = args.path or resolve_path(args.format, args.robot)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        print(f"[robot] loading {args.format.upper()}: {path}")
        self.model = (load_mjcf if args.format == "mjcf" else load_urdf)(path)
        print(f"[robot] {self.model}")

        self.server = viser.ViserServer(host="0.0.0.0", port=args.port)
        try:
            self.server.scene.set_up_direction("+z")  # URDF/MJCF robots are Z-up
        except Exception:
            pass
        self.server.scene.add_grid("/grid", width=4.0, height=4.0, cell_size=0.25,
                                   plane="xy", position=(0.0, 0.0, 0.0))

        self.view = RobotView(self.server, self.model, show_visual=True,
                              show_collision=False)
        self._home = home_pose(self.model)
        self._build_gui()
        self.view.update(self._read_q())

    # -- GUI ------------------------------------------------------------------
    def _build_gui(self):
        with self.server.gui.add_folder("Display"):
            self.g_visual = self.server.gui.add_checkbox("show visual", True)
            self.g_collision = self.server.gui.add_checkbox("show collision", False)
            self.g_animate = self.server.gui.add_checkbox("animate", False)
        self.g_visual.on_update(lambda _: self.view.set_visual_visible(self.g_visual.value))
        self.g_collision.on_update(
            lambda _: self.view.set_collision_visible(self.g_collision.value))

        with self.server.gui.add_folder("Info"):
            self.server.gui.add_text("format", self.args.format)
            self.server.gui.add_text("links", str(len(self.model.links)))
            self.server.gui.add_text("joints", str(len(self.model.joints)))
            self.server.gui.add_text("actuated", str(len(self.model.actuated_joints)))
            self.server.gui.add_text("visual meshes", str(self.view.n_visual))
            self.server.gui.add_text("collision shapes", str(self.view.n_collision))

        self._sliders = {}
        with self.server.gui.add_folder("Joints"):
            self.g_reset = self.server.gui.add_button("reset to home")
            self.g_reset.on_click(lambda _: self._reset())
            for j in self.model.actuated_joints:
                lo, hi = j.lower, j.upper
                if lo >= hi:  # continuous / unlimited
                    lo, hi = -math.pi, math.pi
                s = self.server.gui.add_slider(j.name, float(lo), float(hi),
                                               (hi - lo) / 200.0, float(self._home.get(j.name, 0.0)))
                s.on_update(lambda _: self._pose_changed())
                self._sliders[j.name] = s

    def _read_q(self) -> dict:
        return {name: float(s.value) for name, s in self._sliders.items()}

    def _pose_changed(self):
        if self.g_animate.value:
            return  # animation owns the pose while it's running
        with self._lock:
            self.view.update(self._read_q())

    def _reset(self):
        with self._lock:
            for name, s in self._sliders.items():
                s.value = float(self._home.get(name, 0.0))
            self.view.update(self._read_q())

    # -- main loop ------------------------------------------------------------
    def run(self):
        print("\nviser server running — open the URL above in a browser.\n")
        t0 = time.perf_counter()
        try:
            while True:
                if self.g_animate.value:
                    t = time.perf_counter() - t0
                    q = {}
                    for i, j in enumerate(self.model.actuated_joints):
                        base = self._home.get(j.name, 0.0)
                        amp = 0.35 if ("thigh" in j.name.lower() or "calf" in j.name.lower()) else 0.15
                        q[j.name] = base + amp * math.sin(2.0 * t + i * 0.7)
                    with self._lock:
                        self.view.update(q)
                time.sleep(1.0 / 30.0)
        except KeyboardInterrupt:
            print("\nstopping...")


def main():
    p = argparse.ArgumentParser(description="URDF/MJCF robot viewer (viser)")
    p.add_argument("--format", choices=["mjcf", "urdf"], default="mjcf")
    p.add_argument("--robot", default="go2")
    p.add_argument("--path", default=None, help="explicit model path (overrides --format/--robot)")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    RobotViewer(args).run()


if __name__ == "__main__":
    main()
