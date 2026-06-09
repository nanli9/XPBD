"""GPU spatial-hash broad phase for the 6-DOF solver — the on-device counterpart
of the NumPy ``_grid_candidate_pairs`` grid, built on Warp's ``wp.HashGrid``.

This exists for an apples-to-apples broad-phase comparison. The original grid
(``broadphase="grid"``) is a *uniform spatial hash on the CPU*; the LBVH
(``broadphase="lbvh"``) is a *tree on the GPU*. That confounds two variables at
once — data structure (grid vs tree) **and** where it runs (host vs device).
This module is the missing third corner: a *uniform spatial hash on the GPU*
(``broadphase="hashgrid"``), so:

    grid     → hashgrid   isolates the host round-trip (CPU → GPU, same algorithm)
    hashgrid → lbvh       isolates the data structure   (grid → tree, both on GPU)

``wp.HashGrid.build(points, cell)`` bins body centres into a hash table on device;
``wp.hash_grid_query(grid, p, r)`` walks the cells overlapping a query sphere of
radius ``r`` (correct as long as ``r ≤ cell``, since the 3×3×3 cell neighbourhood
then covers it). Each body queries with a sphere that bounds its inflated AABB
plus the largest other body, then ``emit_pairs_hashgrid`` applies the same exact
per-axis AABB-overlap filter as the LBVH path so the candidate set is identical
in quality. Like the LBVH path it runs before graph replay (not captured), so the
margin can vary per frame for free.
"""

import warp as wp


@wp.func
def aabb_half(q: wp.quat, e: wp.vec3) -> wp.vec3:
    """World-AABB half-size of an OBB: ``|R|·he``."""
    R = wp.quat_to_matrix(q)
    return wp.vec3(
        wp.abs(R[0, 0]) * e[0] + wp.abs(R[0, 1]) * e[1] + wp.abs(R[0, 2]) * e[2],
        wp.abs(R[1, 0]) * e[0] + wp.abs(R[1, 1]) * e[1] + wp.abs(R[1, 2]) * e[2],
        wp.abs(R[2, 0]) * e[0] + wp.abs(R[2, 1]) * e[1] + wp.abs(R[2, 2]) * e[2])


@wp.kernel
def emit_pairs_hashgrid(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    v: wp.array(dtype=wp.vec3),
    omega: wp.array(dtype=wp.vec3),
    he: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    cgroup: wp.array(dtype=int),
    man_margin: float,
    dt: float,
    rs_max: float,                       # largest bounding-sphere radius in the scene
    cell: float,                         # grid cell size (== build radius)
    cap: int,
    # in/out
    count: wp.array(dtype=int),
    pair_a: wp.array(dtype=int),
    pair_b: wp.array(dtype=int),
):
    i = wp.tid()
    e_i = he[i]
    rs_i = wp.length(e_i)
    mi = wp.min(dt * (wp.length(v[i]) + wp.length(omega[i]) * rs_i), rs_i)
    # query sphere: reach any neighbour whose inflated AABB could overlap i's.
    # Clamped to the cell size so the 3×3×3 neighbourhood search stays exact.
    qr = wp.min(rs_i + rs_max + man_margin + mi, cell)
    half_i = aabb_half(q[i], e_i)
    xi = x[i]
    query = wp.hash_grid_query(grid, xi, qr)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        keep = j > i
        if inv_mass[i] == 0.0 and inv_mass[j] == 0.0:
            keep = False
        if cgroup[i] == cgroup[j] and cgroup[i] > 0:
            keep = False
        if keep:
            e_j = he[j]
            rs_j = wp.length(e_j)
            mj = wp.min(dt * (wp.length(v[j]) + wp.length(omega[j]) * rs_j), rs_j)
            half_j = aabb_half(q[j], e_j)
            dpos = xi - x[j]
            pad = man_margin + mi + mj
            ox = wp.abs(dpos[0]) - (half_i[0] + half_j[0] + pad)
            oy = wp.abs(dpos[1]) - (half_i[1] + half_j[1] + pad)
            oz = wp.abs(dpos[2]) - (half_i[2] + half_j[2] + pad)
            if ox <= 0.0 and oy <= 0.0 and oz <= 0.0:
                slot = wp.atomic_add(count, 0, 1)
                if slot < cap:
                    pair_a[slot] = i
                    pair_b[slot] = j
