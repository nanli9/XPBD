"""Greedy constraint-graph coloring for parallel XPBD constraint projection.

Unlike AVBD (which colors *bodies* and runs a per-body local solve), XPBD
projects each *constraint* independently, writing position corrections to the
constraint's endpoints. Two constraints can therefore be solved in parallel iff
they touch **disjoint** particle sets. We color the constraint graph (two
constraints adjacent ⇔ they share a particle); constraints of the same color
write to disjoint particles, so applying their ``Δx`` concurrently is race-free
and exactly equals a Gauss-Seidel sweep in some order. Across colors the sweep
is sequential, which is what gives colored XPBD its faster convergence over the
fully-parallel Jacobi variant (the latter is what the XPBD paper used for its
3D GPU results; we offer both).

The greedy pass is O(E · colors) and never materializes the (potentially dense)
constraint-adjacency graph — a single popular particle shared by ``d``
constraints would otherwise be a ``d``-clique. Instead we track, per particle,
the set of colors already used by constraints touching it, and assign each new
constraint the smallest color free on *both* its endpoints.
"""

from __future__ import annotations

import numpy as np


def color_constraints(
    n_bodies: int,
    c_body_a: np.ndarray,
    c_body_b: np.ndarray,
) -> np.ndarray:
    """Greedily color constraints so same-color constraints share no particle.

    ``c_body_a`` / ``c_body_b`` are int arrays of length ``n_constraints``;
    ``c_body_b[j] < 0`` marks a one-body (world-anchored) constraint. Returns an
    int32 array ``color`` of length ``n_constraints`` with dense color ids
    ``0..k-1``.
    """
    n_c = int(len(c_body_a))
    color = np.full(n_c, -1, dtype=np.int32)
    if n_c == 0:
        return color
    # Per-particle set of colors already taken by an incident constraint.
    body_used: list[set[int]] = [set() for _ in range(n_bodies)]
    # Order by descending degree (Welsh-Powell heuristic) for fewer colors.
    deg = np.zeros(n_bodies, dtype=np.int64)
    for j in range(n_c):
        a = int(c_body_a[j])
        b = int(c_body_b[j])
        if a >= 0:
            deg[a] += 1
        if b >= 0:
            deg[b] += 1
    order = sorted(
        range(n_c),
        key=lambda j: -(int(deg[c_body_a[j]]) + (int(deg[c_body_b[j]]) if c_body_b[j] >= 0 else 0)),
    )
    for j in order:
        a = int(c_body_a[j])
        b = int(c_body_b[j])
        used = set(body_used[a]) if a >= 0 else set()
        if b >= 0:
            used |= body_used[b]
        c = 0
        while c in used:
            c += 1
        color[j] = c
        if a >= 0:
            body_used[a].add(c)
        if b >= 0:
            body_used[b].add(c)
    return color


def color_ranges(color: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Sort constraint indices by color into a flat CSR layout.

    Returns ``(order, starts, num_colors)`` where ``order`` is the constraint
    indices sorted by color, and color ``c`` owns the slice
    ``order[starts[c]:starts[c+1]]``. The solver launches one kernel per color
    over that slice.
    """
    if len(color) == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(1, dtype=np.int32), 0
    num_colors = int(color.max()) + 1
    order = np.argsort(color, kind="stable").astype(np.int32)
    counts = np.bincount(color, minlength=num_colors).astype(np.int32)
    starts = np.zeros(num_colors + 1, dtype=np.int32)
    starts[1:] = np.cumsum(counts)
    return order, starts, num_colors


def color_summary(color: np.ndarray) -> dict[int, int]:
    """Return ``{color_id: count}`` for diagnostics / GUI readouts."""
    out: dict[int, int] = {}
    for c in color:
        out[int(c)] = out.get(int(c), 0) + 1
    return out
