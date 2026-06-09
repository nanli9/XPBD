"""Single particle free fall — confirms the XPBD substep predictor + the
Algorithm-1 velocity update v = (xⁿ⁺¹ − xⁿ)/Δt reproduce analytic gravity."""

import argparse

import numpy as np

from xpbd3d import Solver


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--frames", type=int, default=60)
    args = p.parse_args()

    s = Solver(dt=1 / 60, substeps=10, iterations=1, device=args.device)
    b = s.add_particle((0.0, 0.0, 0.0), mass=1.0, collide=False)
    for _ in range(args.frames):
        s.step()
    t = args.frames / 60.0
    y = s.positions()[b.index, 1]
    vy = s.velocities()[b.index, 1]
    print(f"after {args.frames} frames ({t:.2f}s):")
    print(f"  y  = {y:+.4f}   analytic ½gt² = {0.5 * -9.81 * t * t:+.4f}")
    print(f"  vy = {vy:+.4f}   analytic gt   = {-9.81 * t:+.4f}")
    assert abs(vy - (-9.81 * t)) < 0.05, "velocity update is wrong"
    print("OK")


if __name__ == "__main__":
    main()
