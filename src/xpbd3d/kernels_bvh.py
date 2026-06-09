"""GPU broad phase for the 6-DOF solver: a linear BVH (LBVH) over per-body world
AABBs, built and queried entirely on the device.

This replaces the host round-trip of the spatial-hash grid broad phase (which had
to copy every body's position to the CPU, sort/searchsorted in NumPy, then copy
candidate pairs back each step). Here ``compute_body_aabb`` writes the inflated
world AABBs in place, Warp builds an LBVH from them with the GPU ``lbvh``
constructor (``wp.Bvh(..., constructor='lbvh')`` + ``bvh.rebuild()``), and
``emit_pairs`` queries each body's AABB against the tree and atomically appends
overlapping ``(i, j)`` candidate pairs — no device→host→device copy of the scene.

The AABB inflation is *per body*: each box is padded by the manifold margin plus
its own swept motion this frame (``dt·(|v| + |ω|·r)``), so contacts that form
mid-frame are still caught (Müller 2020 §3.4) without the global velocity term
that used to over-inflate every cell during settling.
"""

import warp as wp


@wp.kernel
def compute_body_aabb(
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    v: wp.array(dtype=wp.vec3),
    omega: wp.array(dtype=wp.vec3),
    he: wp.array(dtype=wp.vec3),
    man_margin: float,
    dt: float,
    # outputs
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
):
    """World-AABB of each OBB (``|R|·he``), inflated by the manifold margin and
    the body's swept motion over the frame so the broad phase is conservative."""
    i = wp.tid()
    R = wp.quat_to_matrix(q[i])
    e = he[i]
    # |R| · he  (column-abs · half-extents) → world-AABB half-size of the OBB.
    ax = wp.abs(R[0, 0]) * e[0] + wp.abs(R[0, 1]) * e[1] + wp.abs(R[0, 2]) * e[2]
    ay = wp.abs(R[1, 0]) * e[0] + wp.abs(R[1, 1]) * e[1] + wp.abs(R[1, 2]) * e[2]
    az = wp.abs(R[2, 0]) * e[0] + wp.abs(R[2, 1]) * e[1] + wp.abs(R[2, 2]) * e[2]
    half = wp.vec3(ax, ay, az)
    rr = wp.length(e)
    m = man_margin + dt * (wp.length(v[i]) + wp.length(omega[i]) * rr)
    pad = wp.vec3(m, m, m)
    lowers[i] = x[i] - half - pad
    uppers[i] = x[i] + half + pad


@wp.kernel
def emit_pairs(
    bvh: wp.uint64,
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    cgroup: wp.array(dtype=int),
    cap: int,
    # in/out
    count: wp.array(dtype=int),          # atomic running pair count (also graph gate)
    pair_a: wp.array(dtype=int),
    pair_b: wp.array(dtype=int),
):
    """For each body, query the LBVH with its (inflated) AABB and append every
    overlapping ``j > i`` as a candidate pair. Drops static-static pairs and
    same-no-self-collide-group pairs (e.g. cloth-vs-itself). Writes are clamped
    to ``cap``; if ``count`` exceeds ``cap`` the host grows + re-emits."""
    i = wp.tid()
    lo = lowers[i]
    hi = uppers[i]
    im = inv_mass[i]
    gi = cgroup[i]
    query = wp.bvh_query_aabb(bvh, lo, hi)
    j = int(0)
    while wp.bvh_query_next(query, j):
        keep = j > i
        if im == 0.0 and inv_mass[j] == 0.0:     # static-static: nothing to solve
            keep = False
        if gi == cgroup[j] and gi > 0:           # same no-self-collide cluster
            keep = False
        if keep:
            slot = wp.atomic_add(count, 0, 1)
            if slot < cap:
                pair_a[slot] = i
                pair_b[slot] = j
