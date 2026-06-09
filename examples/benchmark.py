"""XPBD solver benchmark + profiler.

Measures true device time (warmup + ``wp.synchronize`` around a batch of steps,
so async kernel launches don't under-report) for the demo scenes across solve
modes and devices, and reports throughput in constraint-projections per second.
``--profile`` additionally prints a per-kernel CUDA activity breakdown via
Warp's ScopedTimer.

    uv run python examples/benchmark.py                  # full sweep
    uv run python examples/benchmark.py --profile        # + per-kernel timing
    uv run python examples/benchmark.py --device cpu     # CPU only
"""

import argparse
import time

import numpy as np
import warp as wp

from xpbd3d import Body, Solver, Solver6DOF, Shape


def make_box_pile(nx, ny, nz, device, substeps, iterations):
    s = Solver6DOF(dt=1 / 60, substeps=substeps, iterations=iterations,
                   device=device, floor_y=0.0, friction=0.6)
    h = 0.16
    rng = np.random.default_rng(0)
    sp = 2.25 * h
    for ly in range(ny):
        for ix in range(nx):
            for iz in range(nz):
                s.add_box((ix * sp + rng.uniform(-1e-3, 1e-3), h + ly * sp,
                           iz * sp + rng.uniform(-1e-3, 1e-3)), (h, h, h), mass=1.0)
    return s


def make_cloth(res, device, mode, substeps, iterations):
    s = Solver(dt=1 / 60, substeps=substeps, iterations=iterations,
               device=device, solve_mode=mode, damping=0.01)
    h = 2.4 / (res - 1)
    grid = np.empty((res, res), np.int64)
    for iy in range(res):
        for ix in range(res):
            pinned = iy == 0 and ix in (0, res - 1)
            b = s.add_particle(((ix - res / 2) * h, 2.5, (iy - res / 2) * h),
                               mass=0.0 if pinned else 0.02, collide=False)
            grid[iy, ix] = b.index
    for iy in range(res):
        for ix in range(res):
            if ix + 1 < res:
                s.add_distance(Body(int(grid[iy, ix])), Body(int(grid[iy, ix + 1])), compliance=1e-6)
            if iy + 1 < res:
                s.add_distance(Body(int(grid[iy, ix])), Body(int(grid[iy + 1, ix])), compliance=1e-6)
            if ix + 1 < res and iy + 1 < res:
                s.add_distance(Body(int(grid[iy, ix])), Body(int(grid[iy + 1, ix + 1])), compliance=2e-6)
    return s


def make_stack(n_side, rows, device, mode, substeps, iterations):
    s = Solver(dt=1 / 60, substeps=substeps, iterations=iterations,
               device=device, solve_mode=mode)
    rng = np.random.default_rng(0)
    r = 0.16
    for ly in range(rows):
        for ix in range(n_side):
            for iz in range(n_side):
                b = s.add_particle(((ix - n_side / 2) * 2.3 * r + rng.uniform(-2e-3, 2e-3),
                                    r + ly * 2.3 * r,
                                    (iz - n_side / 2) * 2.3 * r + rng.uniform(-2e-3, 2e-3)),
                                   mass=1.0, shape=Shape("sphere", (r,)), friction=0.4)
                s.add_floor_contact(b, floor_y=r, friction=0.4)
    s.enable_self_collision(True, default_friction=0.4)
    return s


def time_steps(s, frames=120, warmup=20):
    for _ in range(warmup):
        s.step()
    if str(s.device).startswith("cuda"):
        wp.synchronize_device(s.device)
    t0 = time.perf_counter()
    for _ in range(frames):
        s.step()
    if str(s.device).startswith("cuda"):
        wp.synchronize_device(s.device)
    return (time.perf_counter() - t0) / frames * 1000.0  # ms/step


def bench_row(name, s, frames):
    ms = time_steps(s, frames=frames)
    # constraint projections per second = constraints × substeps × iters / step_time
    proj = s.num_constraints * s.substeps * s.iterations
    mhz = proj / (ms / 1000.0) / 1e6
    print(f"  {name:28s} bodies={s.num_bodies:5d} cons={s.num_constraints:6d} "
          f"| {ms:7.2f} ms/step | {1000/ms:6.1f} Hz | {mhz:7.1f} M proj/s")
    return ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="all", help="'cuda:0', 'cpu', or 'all'")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--substeps", type=int, default=15)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()

    devices = []
    if args.device == "all":
        devices = ["cpu"]
        if any(str(d).startswith("cuda") for d in wp.get_devices()):
            devices.append("cuda:0")
    else:
        devices = [args.device]

    for dev in devices:
        print(f"\n=== device: {dev} ===")
        for mode in ("gs", "jacobi"):
            print(f"-- solve mode: {mode} --")
            for res in (16, 32, 48):
                s = make_cloth(res, dev, mode, args.substeps, args.iterations)
                bench_row(f"cloth {res}x{res}", s, args.frames)
            for n_side, rows in ((4, 3), (6, 4), (8, 5)):
                s = make_stack(n_side, rows, dev, mode, args.substeps, args.iterations)
                bench_row(f"stack {n_side}x{n_side}x{rows}", s, args.frames)

        # 6-DOF rigid-body solver (boxes, OBB contacts, CUDA-graph hot loop).
        print("-- 6-DOF rigid boxes (jacobi + graph capture) --")
        sizes = (((3, 3, 3), (5, 4, 5), (8, 6, 8), (12, 10, 12))
                 if str(dev).startswith("cuda") else ((3, 3, 3),))
        for nx, ny, nz in sizes:
            s = make_box_pile(nx, ny, nz, dev, args.substeps, args.iterations)
            for _ in range(30):
                s.step()
            ms = time_steps(s, frames=args.frames)
            print(f"  {'box pile %dx%dx%d' % (nx, ny, nz):28s} boxes={s.num_bodies:5d} "
                  f"pairs={s.n_pairs:5d} | {ms:7.2f} ms/step | {1000/ms:6.1f} Hz")

    if args.profile and any(str(d).startswith("cuda") for d in devices):
        print("\n=== per-kernel CUDA activity (cloth 48x48, jacobi) ===")
        s = make_cloth(48, "cuda:0", "jacobi", args.substeps, args.iterations)
        for _ in range(20):
            s.step()
        wp.synchronize_device("cuda:0")
        try:
            with wp.ScopedTimer("step", cuda_filter=wp.TIMING_KERNEL, print=False) as tm:
                for _ in range(10):
                    s.step()
                wp.synchronize_device("cuda:0")
            wp.timing_print(tm.timing_results)
        except Exception as e:  # older/newer warp timing API differences
            print(f"  (per-kernel profiling unavailable: {e})")


if __name__ == "__main__":
    main()
