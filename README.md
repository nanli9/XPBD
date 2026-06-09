# xpbd3d — 3D Extended Position-Based Dynamics in NVIDIA Warp

A GPU-resident 3D implementation of **XPBD** (Macklin, Müller & Chentanez,
*"XPBD: Position-Based Simulation of Compliant Constrained Dynamics"*, MiG 2016 —
[`reference/XPBD_Macklin2016.pdf`](reference/XPBD_Macklin2016.pdf)) written as
Python kernels on [NVIDIA Warp](https://github.com/NVIDIA/warp), with a
[viser](https://github.com/nerfstudio-project/viser) browser viewer.

It is the XPBD sibling of the AVBD port in `../AVBD`: same tech stack (Warp
kernels + viser viewer + uv), same "run a CUDA scene with live knobs in the
browser" experience. Where AVBD solves an augmented-Lagrangian *vertex block
descent*, this solves the **compliant position-level** XPBD update — the method
whose headline property is that **constraint stiffness is independent of the
iteration count and time step** (paper Fig. 2).

It ships **two solvers**, mirroring `../AVBD`:

- **`Solver`** — 3-DOF point particles + scalar constraints (distance/pin/floor/
  contact). Cloth, chains, particle piles.
- **`Solver6DOF`** — full rigid **boxes**: quaternion orientation, rotated
  inertia, OBB (SAT + face-clip) contacts, friction + restitution
  (Müller et al. 2020). Stacks, dominoes, block piles — with a **CUDA-graph
  captured** hot loop (~5× over per-launch dispatch).

```bash
uv sync && uv pip install -e .

uv run pytest tests/ -q                                   # 22 correctness tests
# 3-DOF
uv run python examples/viewer.py --scene cloth            # the XPBD showcase (CUDA)
uv run python examples/viewer.py --stress                 # 2048-body particle pile
# 6-DOF rigid bodies
uv run python examples/viewer_6dof.py --scene stack       # box tower
uv run python examples/viewer_6dof.py --scene dominoes    # domino cascade
uv run python examples/viewer_6dof.py --stress            # 384-box pile (AVBD parity)
uv run python examples/viewer_6dof.py --mega              # 1440-box pile
uv run python examples/benchmark.py --device cuda:0 --profile
# open the printed URL (default http://localhost:8080)
```

## The algorithm

The solver is a line-by-line transcription of **Algorithm 1**, with
small-substep integration (Müller et al. 2020, *"Detailed Rigid Body Simulation
with XPBD"*) layered on: each frame is split into `substeps` integration steps,
each running `iterations` constraint sweeps, with `λ` reset to 0 per substep.

| Paper | Where |
|---|---|
| predict `x̃ = xⁿ + Δt vⁿ + Δt² M⁻¹ f_ext` (Alg. 1 line 1) | `kernels.integrate` |
| init multipliers `λ₀ ← 0` (Alg. 1 line 4) | `c_lambda.zero_()` per substep in `Solver.step` |
| `Δλ_j = (−C_j − α̃_j λ_j) / (∇C_j M⁻¹ ∇C_jᵀ + α̃_j)` (Eq. 18) | `solve_constraints_color` / `solve_constraints_jacobi` |
| `Δx = M⁻¹ ∇C_jᵀ Δλ_j` (Eq. 17) | same kernels |
| compliance `α̃ = α / Δt²` (§4) | `c_compliance[j] * inv_dt2` |
| update `λ ← λ + Δλ`, `x ← x + Δx` (Alg. 1 lines 9-10) | same kernels |
| velocity `vⁿ⁺¹ = (xⁿ⁺¹ − xⁿ)/Δt` (Alg. 1 line 16) | `kernels.finalize_velocity` |
| zero-compliance contact (§6) | `FLOOR` / `CONTACT` branches (`α = 0`) |
| position-based Coulomb friction | `kernels.friction_delta` (Müller 2020 §3.5) |

`compliance = α` is the XPBD inverse stiffness (m/N); `α = 0` is an infinitely
stiff (hard) PBD constraint. Hard pins are modelled the exact way — a body with
`mass ≤ 0` has `inv_mass = 0` and never moves, so a chain hung from a static
particle stays attached with no constraint at all. `ATTACH` is the *compliant*
pin (a spring to a world point) the paper uses to demonstrate stiffness control.

### Two solve modes — both from the paper

* **`jacobi`** (default on CUDA) — the paper's 3D GPU mode (§6): every constraint
  is projected in one launch, corrections accumulated with atomics and averaged
  (`solve_constraints_jacobi` + `apply_jacobi`). No graph coloring, so dynamic
  contact sets cost nothing to maintain — fully parallel.
* **`gs`** (default on CPU) — colored Gauss-Seidel: constraints are graph-colored
  (`coloring.color_constraints`) so a color class touches disjoint particles;
  one launch per color, parallel within a color and sequential across colors.
  Better convergence per iteration, recolored only when the constraint set
  changes.

Constraint types (`kernels.py`): `DISTANCE` (springs / cloth / chain links,
per-constraint compliance), `ATTACH` (compliant pin), `FLOOR` (one-sided
push-only + friction), `CONTACT` (one-sided sphere-sphere + friction).

## Scenes & knobs (`examples/viewer.py`)

| scene | what | notable |
|---|---|---|
| `chain` | N-link chain on a static anchor + droppable spheres | stretch with the stiffness slider |
| `cloth` | a draped sheet pinned at two corners (structural + shear + bend) | **the XPBD showcase** — soft↔stiff at fixed iterations |
| `stack` / `--stress` / `--mega` | sphere pile on a floor with friction (384 / 2048 bodies) | the self-collision + GPU benchmark |

Live GUI knobs: **substeps** (the "small steps" stability dial), **iterations /
substep**, **gravity**, **friction μ**, **solve mode** (jacobi/gs), and the
headline **log₁₀ compliance** slider — drag it and the same scene goes from rigid
to soft *without* changing the iteration count. A Performance panel reports
device, bodies, constraints, colors/mode, step time, solver capacity and the
wall tick rate.

## Performance

Measured on an **RTX 3060 Laptop GPU**, 15 substeps × 2 iterations, true device
time (warmup + `wp.synchronize`). `M proj/s` = constraint projections per second
(`constraints × substeps × iterations / step_time`).

| scene | bodies | cons | gs | **jacobi** | jacobi rate |
|---|--:|--:|--:|--:|--:|
| cloth 16² | 256 | 705 | 6.6 ms | **2.4 ms (418 Hz)** | 8.8 M/s |
| cloth 32² | 1024 | 2945 | 6.5 ms | **2.3 ms (440 Hz)** | 38.8 M/s |
| cloth 48² | 2304 | 6721 | 6.5 ms | **2.3 ms (441 Hz)** | 88.8 M/s |
| stack 8×8×5 | 320 | 576 | 7.8 ms | **5.2 ms (193 Hz)** | 3.3 M/s |
| `--stress` 8×8×6 | 384 | ~700 | — | **4.8 ms (209 Hz)** | — |
| `--mega` 16×16×8 | 2048 | ~3.8k | — | **16.2 ms (62 Hz)** | — |

The cloth path is **fully GPU-resident** (constraint set is static, so it's
flushed once) — the per-frame hot loop is pure Warp launches with no host
readbacks, hence the 440 Hz at 2300 bodies. Self-collision scenes do one host
readback per frame for the broad phase, which uses a **uniform spatial-hash
grid** (`solver._grid_candidate_pairs`, O(N)); replacing the original dense
O(N²) sweep took the 1152-body pile from 58 ms → 11 ms and lets `--mega` (2048
bodies) run at 62 Hz. Per-kernel CUDA timing is available via
`benchmark.py --profile`.

> Like AVBD's 3-DOF particle solver, the broad phase runs on the host (the grid
> is cheap and keeps the substep loop GPU-resident). A fully GPU broad phase
> (LBVH) is the natural next step for piles beyond a few thousand bodies.

## 6-DOF rigid bodies (`Solver6DOF`)

A faithful transcription of **Müller et al. 2020, Algorithm 2**
([`reference/Mueller2020_RigidBodyXPBD.pdf`](reference/Mueller2020_RigidBodyXPBD.pdf)).
Each substep: integrate `x`/`q` and `v`/`ω` (with optional gyroscopic term),
solve positional contacts, derive velocities from `(x−x_prev)/h` and `Δq`, then a
velocity pass for dynamic friction + restitution.

| Paper | Where |
|---|---|
| integrate `x`,`q`,`v`,`ω` (Alg. 2) | `kernels_6dof.integrate_bodies` |
| generalized inverse mass `w = 1/m + (r×n)ᵀI⁻¹(r×n)` (Eq. 2/3) | `gen_inv_mass` |
| `Δλ = (−c − α̃λ)/(w₁+w₂+α̃)`, `Δx = Δλ n` (Eq. 4-9) | `solve_box_manifold`, `solve_floor_contacts` |
| world inverse inertia `R I⁻¹_body Rᵀ` | `world_invI_mul` |
| OBB contact manifold (SAT + Sutherland-Hodgman clip) | `generate_box_manifold` |
| velocity pass: friction Eq. 30 + restitution (§3.6) | `velocity_box`, `velocity_floor` |
| static friction at position level (§3.5) | in `solve_floor_contacts` |

Contacts use the paper's **Jacobi** projection (atomic accumulate + averaged
apply) — no graph coloring. Box-box collisions use a face-axis SAT normal with a
clipped face manifold (up to 8 points/pair), generated once per substep and
frozen for the solve; corner-vs-OBB is not enough (aligned stacks would collapse
since a top box's corners sit exactly on the lower box's edges). Edge-edge
contacts are the known gap, as in AVBD's notes.

### Performance & the optimization story

Measured on an **RTX 3060 Laptop**, 15 substeps × 1 iteration, true device time.
The headline optimization is **CUDA-graph capture** of the fixed substep-loop
kernel sequence (~150 launches/frame collapse to one graph replay):

| box pile | boxes | baseline | **graph capture** | speedup |
|---|--:|--:|--:|--:|
| 3×3×3 | 27 | 5.46 ms | **1.34 ms (745 Hz)** | 4.1× |
| 5×4×5 | 100 | 5.73 ms | **1.56 ms (640 Hz)** | 3.7× |
| 8×6×8 | 384 | 7.00 ms | **2.50 ms (399 Hz)** | 2.8× |
| 12×10×12 | 1440 | — | **5.10 ms (196 Hz)** | — |

Profiling drove the order of attack (see `benchmark.py --profile`):

1. The flat ~5.5 ms floor at small N was **launch overhead** — 3.4 ms of 5.7 ms
   at 100 boxes was pure CPU dispatch for ~150 kernels/frame. **CUDA-graph
   capture** (fixed launch dims via a device-side pair count) removed it.
2. The next bottleneck became the **host broad phase**; the dict-loop grid was
   replaced with a fully-vectorised sort + `searchsorted` cell hash (~6×), and
   the bounding-sphere test with a tight **world-AABB** overlap (it dropped the
   384-box candidate set from 3662 phantom pairs to the 320 real contacts).
3. What remains is genuinely **per-body GPU compute** (the contact + integrate
   kernels) plus the O(N) host broad phase — both well-balanced, so further
   micro-opts are noise. A fully-GPU broad phase (LBVH) is the next large lever
   for piles beyond a few thousand boxes.

The math is unchanged throughout — these are pure implementation optimizations.

## Layout

```
src/xpbd3d/
├── solver.py        # Solver (3-DOF): scene, substep loop, vectorised grid broad phase
├── kernels.py       # 3-DOF Warp kernels: integrate, GS + Jacobi solves, friction
├── solver_6dof.py   # Solver6DOF: rigid-box scene, AABB broad phase, CUDA-graph step
├── kernels_6dof.py  # 6-DOF kernels: integrate, OBB SAT manifold, contact + velocity solve
├── coloring.py      # greedy constraint-graph coloring (3-DOF GS mode)
└── scene.py         # Body / Shape / ConstraintHandle handles
examples/
├── viewer.py        # 3-DOF viser viewer (chain / cloth / stack)
├── viewer_6dof.py   # 6-DOF viser viewer (stack / dominoes / stress / mega)
├── cloth_drop.py    # headless cloth drape (--plot saves a 3D snapshot)
├── hanging_chain.py # headless chain (--plot saves a PNG)
├── smoke_test.py    # single-particle free fall sanity
└── benchmark.py     # device timing sweep (3-DOF + 6-DOF) + per-kernel profiler
tests/
├── test_solver.py       # 14 tests (incl. compliance-iteration-independence, Fig. 2)
└── test_solver_6dof.py  # 8 tests (free fall, floor rest, spin, stack, pile, tip, dominoes)
reference/
├── XPBD_Macklin2016.pdf            # the 3-DOF / compliant-constraint paper
└── Mueller2020_RigidBodyXPBD.pdf   # the 6-DOF rigid-body paper
```

## Notes & sharp edges

- **Velocity-aware contact margin.** Contacts are rebuilt once per frame but
  solved across all substeps, so the broad phase pads the trigger radius by one
  frame of relative motion (capped at one body radius) — otherwise a body
  crosses the contact threshold mid-frame, penetrates freely and gets an
  explosive separation impulse next frame. This is the Müller 2020 §3.4 fix and
  is what makes the stacks settle to *zero* penetration.
- **Friction is position-based** (Müller 2020 §3.5), a square-region Coulomb
  cone limiting the per-substep tangential slide to `μ·d`. No restitution pass,
  so contacts are near-inelastic (correct for settling piles).
- **3-DOF point masses only.** Bodies are particles; cubes/pillars are visual,
  and collision uses the inscribed sphere. Full 6-DOF rigid bodies (quaternion
  orientation, OBB contact) are the analog of `../AVBD`'s `Solver6DOF` and are
  not ported here.
- **Apple Silicon ⇒ CPU-only Warp**; pass `--device cpu`. All kernels are
  GPU-clean and run unchanged on CUDA.
