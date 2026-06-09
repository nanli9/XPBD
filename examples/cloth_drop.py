"""Cloth draping — the XPBD showcase.

A square sheet pinned at its two top corners sags under gravity. Sweeping the
compliance shows soft → stiff drape at a *fixed* iteration count (the property
PBD lacks). Prints the centre-sag depth; ``--plot`` saves a 3D snapshot.

    uv run python examples/cloth_drop.py --res 30 --device cuda:0
    uv run python examples/cloth_drop.py --res 30 --compliance 1e-3 --plot
"""

import argparse

import numpy as np

from xpbd3d import Body, Solver, Shape


def build_cloth(res, size, mass, compliance, device, substeps, iterations):
    s = Solver(dt=1 / 60, substeps=substeps, iterations=iterations,
               device=device, solve_mode="auto", damping=0.01)
    h = size / (res - 1)
    grid = np.empty((res, res), dtype=np.int64)
    for iy in range(res):
        for ix in range(res):
            pinned = (iy == 0 and ix in (0, res - 1))
            b = s.add_particle(((ix - (res - 1) / 2) * h, 2.5, (iy - (res - 1) / 2) * h),
                               mass=0.0 if pinned else mass, collide=False)
            grid[iy, ix] = b.index

    def link(a, b, c):
        s.add_distance(Body(int(a)), Body(int(b)), compliance=c)

    for iy in range(res):
        for ix in range(res):
            if ix + 1 < res:
                link(grid[iy, ix], grid[iy, ix + 1], compliance)
            if iy + 1 < res:
                link(grid[iy, ix], grid[iy + 1, ix], compliance)
            if ix + 1 < res and iy + 1 < res:
                link(grid[iy, ix], grid[iy + 1, ix + 1], compliance * 2)
                link(grid[iy, ix + 1], grid[iy + 1, ix], compliance * 2)
            if ix + 2 < res:
                link(grid[iy, ix], grid[iy, ix + 2], compliance * 4)
            if iy + 2 < res:
                link(grid[iy, ix], grid[iy + 2, ix], compliance * 4)
    return s, grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--res", type=int, default=24)
    p.add_argument("--size", type=float, default=2.4)
    p.add_argument("--mass", type=float, default=0.02)
    p.add_argument("--compliance", type=float, default=1e-6)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--substeps", type=int, default=15)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    s, grid = build_cloth(args.res, args.size, args.mass, args.compliance,
                          args.device, args.substeps, args.iterations)
    print(f"cloth {args.res}x{args.res}: {s.num_bodies} particles, "
          f"{s.num_constraints} constraints, mode={s.solve_mode}")
    for _ in range(args.frames):
        s.step()
    pos = s.positions()
    centre = pos[grid[args.res // 2, args.res // 2]]
    print(f"settled: centre y = {centre[1]:.3f}  (sag = {2.5 - centre[1]:.3f} m)")

    if args.plot:
        import matplotlib.pyplot as plt
        verts = pos[grid.reshape(-1)]
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(verts[:, 0], verts[:, 2], verts[:, 1],
                        color="#5588dd", alpha=0.85, edgecolor="none")
        ax.set_title(f"XPBD cloth (α={args.compliance:g})")
        ax.view_init(elev=18, azim=-60)
        fig.savefig("cloth_drop.png", dpi=120, bbox_inches="tight")
        print("wrote cloth_drop.png")


if __name__ == "__main__":
    main()
