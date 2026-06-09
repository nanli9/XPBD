"""Correctness tests for the 6-DOF rigid-body XPBD solver (Müller et al. 2020).

Defaults to CUDA if available (the graph-captured path), else CPU. Override with
``XPBD_TEST_DEVICE``.
"""

import os

import numpy as np
import pytest
import warp as wp

from xpbd3d import Solver6DOF

wp.init()
_HAS_CUDA = any(str(d).startswith("cuda") for d in wp.get_devices())
DEVICE = os.environ.get("XPBD_TEST_DEVICE", "cuda:0" if _HAS_CUDA else "cpu")


def test_free_fall():
    s = Solver6DOF(dt=1 / 60, substeps=10, iterations=1, device=DEVICE, floor_y=-100.0)
    s.add_box((0, 0, 0), (0.2, 0.2, 0.2), mass=1.0)
    for _ in range(60):
        s.step()
    assert abs(s.positions()[0, 1] - 0.5 * -9.81) < 0.05
    assert abs(s.velocities()[0, 1] - (-9.81)) < 0.05


def test_box_rests_on_floor():
    s = Solver6DOF(dt=1 / 60, substeps=15, iterations=1, device=DEVICE,
                   floor_y=0.0, friction=0.6)
    s.add_box((0, 1.0, 0), (0.2, 0.2, 0.2), mass=1.0)
    for _ in range(180):
        s.step()
    p = s.positions()[0]
    q = s.orientations()[0]
    assert abs(p[1] - 0.2) < 2e-3                 # rests at half-extent
    assert abs(s.velocities()[0, 1]) < 1e-2
    assert abs(abs(q[3]) - 1.0) < 1e-2            # no spurious rotation (q ≈ identity)


def test_spin_conservation():
    s = Solver6DOF(dt=1 / 60, substeps=10, iterations=1, device=DEVICE,
                   gravity=(0, 0, 0), floor_y=-100.0)
    s.add_box((0, 0, 0), (0.2, 0.1, 0.3), mass=1.0)
    s._flush()
    om = s.omega.numpy().copy(); om[0] = [0, 5.0, 0]
    s.omega = wp.array(om, dtype=wp.vec3, device=DEVICE)
    s._graph = None
    for _ in range(120):
        s.step()
    w = s.omega.numpy()[0]
    assert abs(w[1] - 5.0) < 0.5                  # principal-axis spin preserved
    assert np.linalg.norm(w[[0, 2]]) < 0.5


@pytest.mark.parametrize("n", [3, 5])
def test_stack_settles_no_penetration(n):
    s = Solver6DOF(dt=1 / 60, substeps=20, iterations=1, device=DEVICE,
                   floor_y=0.0, friction=0.6)
    h = 0.2
    for k in range(n):
        s.add_box((0, h + k * (2 * h + 0.02), 0), (h, h, h), mass=1.0)
    for _ in range(350):
        s.step()
    ys = np.sort(s.positions()[:, 1])
    assert np.all(np.isfinite(s.positions()))
    assert ys.max() < n * (2 * h) + 0.2          # did not explode
    # consecutive boxes sit ~2h apart (clean stack, no penetration)
    assert np.allclose(np.diff(ys), 2 * h, atol=8e-3)
    assert np.abs(s.velocities()).max() < 0.1     # at rest


def test_pile_stable():
    s = Solver6DOF(dt=1 / 60, substeps=18, iterations=1, device=DEVICE,
                   floor_y=0.0, friction=0.6)
    h = 0.16
    rng = np.random.default_rng(0)
    for ly in range(3):
        for ix in range(3):
            for iz in range(3):
                s.add_box((ix * 0.34 + rng.uniform(-2e-3, 2e-3), h + ly * 0.34,
                           iz * 0.34 + rng.uniform(-2e-3, 2e-3)), (h, h, h), mass=1.0)
    for _ in range(400):
        s.step()
    p = s.positions()
    assert np.all(np.isfinite(p))
    assert p[:, 1].max() < 1.2                    # 3 layers, stayed a pile
    assert np.abs(s.velocities()).max() < 0.2


def test_box_tips_and_rests_flat():
    s = Solver6DOF(dt=1 / 60, substeps=20, iterations=1, device=DEVICE,
                   floor_y=0.0, friction=0.7)
    hw, hh, hd = 0.05, 0.25, 0.12
    s.add_box((0, hh, 0), (hw, hh, hd), mass=1.0)
    s._flush()
    om = s.omega.numpy().copy(); om[0] = [0, 0, -3.0]
    s.omega = wp.array(om, dtype=wp.vec3, device=DEVICE)
    s._graph = None
    for _ in range(300):
        s.step()
    p = s.positions()[0]
    assert abs(p[1] - hw) < 0.03                  # toppled, resting on its side
    assert np.linalg.norm(s.omega.numpy()[0]) < 0.2


def test_domino_cascade():
    s = Solver6DOF(dt=1 / 60, substeps=20, iterations=1, device=DEVICE,
                   floor_y=0.0, friction=0.6)
    hw, hh, hd = 0.04, 0.22, 0.12
    sp = 0.18
    n = 6
    q0 = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -0.45)
    for k in range(n):
        q = (q0[0], q0[1], q0[2], q0[3]) if k == 0 else (0, 0, 0, 1)
        y = hh * 0.9 if k == 0 else hh
        s.add_box((k * sp, y, 0.0), (hw, hh, hd), mass=1.0, quaternion=q)
    for _ in range(420):
        s.step()
    fell = int(np.sum(s.positions()[:, 1] < hh * 0.6))
    assert fell >= n - 1                          # the cascade propagated
