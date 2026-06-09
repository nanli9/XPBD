"""Correctness tests for the XPBD 3D particle solver.

Run on CPU by default (portable); set ``XPBD_TEST_DEVICE=cuda:0`` to test on GPU.
The marquee test is ``test_compliance_iteration_independent`` — the property
that XPBD exists to provide (paper Fig. 2).
"""

import math
import os

import numpy as np
import pytest

from xpbd3d import Body, Solver, Shape
from xpbd3d.coloring import color_constraints

DEVICE = os.environ.get("XPBD_TEST_DEVICE", "cpu")


def test_free_fall_matches_analytic():
    s = Solver(dt=1 / 60, substeps=10, iterations=1, device=DEVICE)
    b = s.add_particle((0, 0, 0), mass=1.0, collide=False)
    for _ in range(60):
        s.step()
    t = 1.0
    assert abs(s.velocities()[b.index, 1] - (-9.81 * t)) < 0.05
    assert abs(s.positions()[b.index, 1] - 0.5 * -9.81 * t * t) < 0.05


def test_static_body_never_moves():
    s = Solver(dt=1 / 60, substeps=10, device=DEVICE)
    a = s.add_particle((0, 1, 0), mass=0.0, collide=False)
    for _ in range(30):
        s.step()
    assert np.allclose(s.positions()[a.index], [0, 1, 0])


@pytest.mark.parametrize("mode", ["gs", "jacobi"])
def test_hard_distance_holds_length(mode):
    s = Solver(dt=1 / 60, substeps=15, iterations=2, device=DEVICE, solve_mode=mode)
    a = s.add_particle((0, 2, 0), mass=0.0, collide=False)
    b = s.add_particle((0.5, 2, 0), mass=1.0, collide=False)
    s.add_distance(a, b, rest=0.5, compliance=0.0)
    for _ in range(120):
        s.step()
    L = np.linalg.norm(s.positions()[b.index] - s.positions()[a.index])
    assert abs(L - 0.5) < 1e-3


@pytest.mark.parametrize("iters", [1, 4, 16])
def test_compliance_iteration_independent(iters):
    """The XPBD headline (Fig. 2): a compliant spring's steady-state stretch is
    set by α (= m·g·α), independent of the iteration count."""
    alpha = 1e-3
    s = Solver(dt=1 / 60, substeps=20, iterations=iters, device=DEVICE,
               solve_mode="gs", damping=0.05)
    a = s.add_particle((0, 2, 0), mass=0.0, collide=False)
    b = s.add_particle((0, 1, 0), mass=1.0, collide=False)
    s.add_distance(a, b, rest=1.0, compliance=alpha)
    for _ in range(800):
        s.step()
    stretch = np.linalg.norm(s.positions()[b.index] - s.positions()[a.index]) - 1.0
    analytic = 1.0 * 9.81 * alpha  # F = k x, k = 1/alpha, F = m g
    assert abs(stretch - analytic) < 0.1 * analytic


def test_floor_contact_rest():
    s = Solver(dt=1 / 60, substeps=10, iterations=2, device=DEVICE)
    b = s.add_particle((0, 2, 0), mass=1.0, shape=Shape("sphere", (0.1,)), collide=False)
    s.add_floor_contact(b, floor_y=0.1, friction=0.5)
    for _ in range(180):
        s.step()
    assert abs(s.positions()[b.index, 1] - 0.1) < 1e-3
    assert abs(s.velocities()[b.index, 1]) < 1e-2


@pytest.mark.parametrize("mode", ["gs", "jacobi"])
def test_two_spheres_do_not_overlap(mode):
    s = Solver(dt=1 / 60, substeps=15, iterations=2, device=DEVICE, solve_mode=mode)
    a = s.add_particle((0.0, 1.0, 0), mass=1.0, shape=Shape("sphere", (0.2,)))
    b = s.add_particle((0.15, 0.4, 0), mass=1.0, shape=Shape("sphere", (0.2,)))
    s.add_floor_contact(a, floor_y=0.2, friction=0.4)
    s.add_floor_contact(b, floor_y=0.2, friction=0.4)
    s.enable_self_collision(True, default_friction=0.4)
    for _ in range(240):
        s.step()
    d = np.linalg.norm(s.positions()[a.index] - s.positions()[b.index])
    assert d >= 0.4 - 2e-3  # radii sum, allow µm tolerance


@pytest.mark.parametrize("mode", ["gs", "jacobi"])
def test_stack_column_settles(mode):
    """A vertical column of spheres must settle to a clean stack (the bug that
    the velocity-aware broad-phase margin fixes)."""
    s = Solver(dt=1 / 60, substeps=20, iterations=2, device=DEVICE,
               solve_mode=mode, max_speed=0.0)
    r = 0.16
    for k in range(4):
        b = s.add_particle((0, r + k * 2.1 * r, 0), mass=1.0,
                           shape=Shape("sphere", (r,)), friction=0.3)
        s.add_floor_contact(b, floor_y=r, friction=0.3)
    s.enable_self_collision(True, margin=0.05, default_friction=0.3)
    for _ in range(220):
        s.step()
    ys = np.sort(s.positions()[:, 1])
    assert ys.max() < 1.5  # did not explode
    assert np.all(np.isfinite(s.positions()))
    # neighbours sit ~2r apart
    assert np.allclose(np.diff(ys), 2 * r, atol=5e-3)


def test_friction_decelerates_slide():
    s = Solver(dt=1 / 60, substeps=10, iterations=2, device=DEVICE)
    b = s.add_particle((0, 0.1, 0), mass=1.0, shape=Shape("sphere", (0.1,)), collide=False)
    s.add_floor_contact(b, floor_y=0.1, friction=0.5)
    s.set_velocity(b, (3.0, 0, 0))
    for _ in range(120):
        s.step()
    assert abs(s.velocities()[b.index, 0]) < 0.5  # slowed substantially


def test_coloring_is_valid():
    """No two same-color constraints may share a body."""
    rng = np.random.default_rng(0)
    n_bodies = 40
    a = rng.integers(0, n_bodies, 200)
    b = rng.integers(0, n_bodies, 200)
    # Real constraints always join two *distinct* bodies (no self-loops).
    b = np.where(b == a, (b + 1) % n_bodies, b)
    color = color_constraints(n_bodies, a, b)
    # explicit pairwise check among constraints sharing a body
    incident = {}
    for j in range(len(a)):
        for body in (int(a[j]), int(b[j])):
            incident.setdefault(body, []).append(j)
    for body, js in incident.items():
        cols = [int(color[j]) for j in js]
        assert len(cols) == len(set(cols)), f"body {body} has same-color constraints"
