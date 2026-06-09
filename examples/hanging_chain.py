"""N-link hanging chain (the classic compliant-constraint demo).

Hung from a static anchor, the chain settles to a vertical line; with
``--compliance`` you can make the links stretchy and watch the steady-state
elongation scale with α — independently of the iteration count (XPBD Fig. 2).

    uv run python examples/hanging_chain.py --n 12 --device cuda:0
    uv run python examples/hanging_chain.py --n 12 --compliance 1e-3 --plot
"""

import argparse

import numpy as np

from xpbd3d import Solver, Shape


def build(n, link, compliance, device, substeps, iterations, heavy):
    s = Solver(dt=1 / 60, substeps=substeps, iterations=iterations,
               device=device, solve_mode="gs")
    anchor = s.add_particle((0.0, 0.0, 0.0), mass=0.0, collide=False)
    prev = anchor
    for i in range(1, n + 1):
        b = s.add_particle((i * link, 0.0, 0.0), mass=1.0, collide=False)
        s.add_distance(prev, b, rest=link, compliance=compliance)
        prev = b
    if heavy > 0:
        b = s.add_particle(((n + 1) * link, 0.0, 0.0), mass=heavy, collide=False)
        s.add_distance(prev, b, rest=link, compliance=compliance)
    return s, anchor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--link", type=float, default=0.3)
    p.add_argument("--compliance", type=float, default=0.0)
    p.add_argument("--heavy", type=float, default=0.0)
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--substeps", type=int, default=15)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    s, anchor = build(args.n, args.link, args.compliance, args.device,
                      args.substeps, args.iterations, args.heavy)
    for _ in range(args.frames):
        s.step()
    pos = s.positions()
    # Measure each link's stretch relative to rest.
    stretches = []
    for i in range(args.n):
        d = np.linalg.norm(pos[i + 1] - pos[i])
        stretches.append(d - args.link)
    print(f"chain settled: lowest y = {pos[:, 1].min():.3f}")
    print(f"mean link stretch = {np.mean(stretches) * 1000:.3f} mm "
          f"(compliance α = {args.compliance:g})")

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 6))
        ax.plot(pos[:, 0], pos[:, 1], "o-", color="#3a5fd0")
        ax.scatter([pos[0, 0]], [pos[0, 1]], color="red", zorder=5, label="anchor")
        ax.set_aspect("equal")
        ax.set_title(f"XPBD hanging chain (n={args.n}, α={args.compliance:g})")
        ax.legend()
        fig.savefig("hanging_chain.png", dpi=120, bbox_inches="tight")
        print("wrote hanging_chain.png")


if __name__ == "__main__":
    main()
