"""Warp kernels for the 6-DOF rigid-body XPBD solver.

Implements Müller, Macklin, Chentanez, Jeschke & Kim 2020, *"Detailed Rigid Body
Simulation with Extended Position Based Dynamics"* (Computer Graphics Forum) —
see ``reference/Mueller2020_RigidBodyXPBD.pdf``. Equation numbers below are from
that paper.

State per body: position ``x`` (vec3), orientation ``q`` (unit quat, xyzw),
linear velocity ``v``, angular velocity ``omega`` (world frame), inverse mass,
and body-frame diagonal inverse inertia ``inv_I`` (vec3). Static bodies have
``inv_mass = 0`` and ``inv_I = 0``.

Contacts use a **corner-vs-OBB** manifold (each box corner tested against the
floor and against the other box's faces) — robust for resting/stacking/domino
scenes; edge-edge contacts are the known gap, as in AVBD's notes. All contact
corrections are accumulated into per-body Jacobi buffers (`dx`, `drot`, count)
via atomics and applied averaged — the paper's order-independent parallel solve
(§3.3) with no per-frame graph coloring, exactly as requested for the GPU.
"""

import warp as wp

EPS = wp.constant(1.0e-9)

# Collision shape types (per body). A sphere is a capsule with zero segment
# half-length. Capsules and cylinders both extend along the body-local +Z axis
# and share the body-body narrow phase (segment + radius); they differ at the
# floor, where a CYLINDER uses its flat caps / rim exactly (solve_floor_cylinder)
# while a CAPSULE uses rounded ends.
SHAPE_BOX = wp.constant(0)
SHAPE_CAPSULE = wp.constant(1)
SHAPE_CYLINDER = wp.constant(2)


# -----------------------------------------------------------------------------
# Quaternion / inertia helpers
# -----------------------------------------------------------------------------
@wp.func
def box_corner(he: wp.vec3, i: int) -> wp.vec3:
    """Body-frame position of box corner ``i`` (0..7) from the ±half-extents."""
    sx = wp.where((i & 1) != 0, 1.0, -1.0)
    sy = wp.where((i & 2) != 0, 1.0, -1.0)
    sz = wp.where((i & 4) != 0, 1.0, -1.0)
    return wp.vec3(sx * he[0], sy * he[1], sz * he[2])


@wp.func
def world_invI_mul(q: wp.quat, inv_I: wp.vec3, u: wp.vec3) -> wp.vec3:
    """Apply the world-frame inverse inertia to vector ``u``:
    ``I⁻¹_world u = R (I⁻¹_body ⊙ (Rᵀ u))`` with ``R = q``."""
    ub = wp.quat_rotate_inv(q, u)
    ub = wp.vec3(ub[0] * inv_I[0], ub[1] * inv_I[1], ub[2] * inv_I[2])
    return wp.quat_rotate(q, ub)


@wp.func
def gen_inv_mass(inv_m: float, q: wp.quat, inv_I: wp.vec3,
                 r: wp.vec3, n: wp.vec3) -> float:
    """Generalized inverse mass for a unit correction ``n`` at body offset ``r``
    (Eqs. 2/3): ``w = 1/m + (r×n)ᵀ I⁻¹ (r×n)``."""
    rn = wp.cross(r, n)
    return inv_m + wp.dot(rn, world_invI_mul(q, inv_I, rn))


@wp.func
def quat_integrate(q: wp.quat, omega: wp.vec3, h: float) -> wp.quat:
    """Linearized quaternion integration ``q ← q + h·½[ω,0]·q`` then normalize."""
    wq = wp.quat(omega[0], omega[1], omega[2], 0.0)
    q = q + (0.5 * h) * (wq * q)
    return wp.normalize(q)


# -----------------------------------------------------------------------------
# Integration (Algorithm 2, first inner loop)
# -----------------------------------------------------------------------------
@wp.kernel
def integrate_bodies(
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    v: wp.array(dtype=wp.vec3),
    omega: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    inv_I: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
    h: float,
    gyroscopic: int,
    # outputs
    x_prev: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.quat),
):
    i = wp.tid()
    x_prev[i] = x[i]
    q_prev[i] = q[i]
    if inv_mass[i] == 0.0:
        return
    # linear
    v[i] = v[i] + gravity * h
    x[i] = x[i] + v[i] * h
    # angular: ω += h I⁻¹(τ_ext − ω × (I ω)); τ_ext = 0
    if gyroscopic != 0:
        # I_world ω via the inverse of inv_I (diagonal): I_body = 1/inv_I
        wbody = wp.quat_rotate_inv(q[i], omega[i])
        ii = inv_I[i]
        ix = wp.where(ii[0] > 0.0, 1.0 / ii[0], 0.0)
        iy = wp.where(ii[1] > 0.0, 1.0 / ii[1], 0.0)
        iz = wp.where(ii[2] > 0.0, 1.0 / ii[2], 0.0)
        Iw = wp.vec3(ix * wbody[0], iy * wbody[1], iz * wbody[2])
        tau = -wp.cross(wbody, Iw)              # body-frame gyroscopic torque
        dwb = wp.vec3(ii[0] * tau[0], ii[1] * tau[1], ii[2] * tau[2]) * h
        omega[i] = omega[i] + wp.quat_rotate(q[i], dwb)
    q[i] = quat_integrate(q[i], omega[i], h)


# -----------------------------------------------------------------------------
# Jacobi accumulation helpers — applied to a body pair (A positive, B negative)
# -----------------------------------------------------------------------------
@wp.func
def accumulate_correction(
    dx: wp.array(dtype=wp.vec3),
    drot: wp.array(dtype=wp.vec3),
    dcount: wp.array(dtype=int),
    a: int, b: int,
    inv_m_a: float, inv_m_b: float,
    qa: wp.quat, qb: wp.quat,
    inv_I_a: wp.vec3, inv_I_b: wp.vec3,
    ra: wp.vec3, rb: wp.vec3,
    p: wp.vec3,
):
    """Distribute positional impulse ``p`` to body A (+) and body B (−) — the
    linear part to ``dx`` and the angular rotvec ``I⁻¹(r×p)`` to ``drot`` (Eqs.
    6-9). B may be ``-1`` (static world)."""
    if inv_m_a > 0.0:
        wp.atomic_add(dx, a, p * inv_m_a)
        wp.atomic_add(drot, a, world_invI_mul(qa, inv_I_a, wp.cross(ra, p)))
        wp.atomic_add(dcount, a, 1)
    if b >= 0 and inv_m_b > 0.0:
        wp.atomic_add(dx, b, -p * inv_m_b)
        wp.atomic_add(drot, b, -world_invI_mul(qb, inv_I_b, wp.cross(rb, p)))
        wp.atomic_add(dcount, b, 1)


@wp.func
def obb_penetration(p_world: wp.vec3, xb: wp.vec3, qb: wp.quat, he_b: wp.vec3):
    """Test world point ``p_world`` against box B. Returns ``(inside, s, d)``
    where ``s`` is the outward separation normal (world, B→point) of the nearest
    face and ``d`` the penetration depth (>0 if inside)."""
    local = wp.quat_rotate_inv(qb, p_world - xb)
    px = he_b[0] - wp.abs(local[0])
    py = he_b[1] - wp.abs(local[1])
    pz = he_b[2] - wp.abs(local[2])
    inside = (px > 0.0) and (py > 0.0) and (pz > 0.0)
    # nearest face = smallest penetration axis
    d = px
    n_local = wp.vec3(wp.where(local[0] >= 0.0, 1.0, -1.0), 0.0, 0.0)
    if py < d:
        d = py
        n_local = wp.vec3(0.0, wp.where(local[1] >= 0.0, 1.0, -1.0), 0.0)
    if pz < d:
        d = pz
        n_local = wp.vec3(0.0, 0.0, wp.where(local[2] >= 0.0, 1.0, -1.0))
    return inside, wp.quat_rotate(qb, n_local), d


# -----------------------------------------------------------------------------
# Floor contacts (per box: 8 corners vs an infinite plane y = floor_y)
# -----------------------------------------------------------------------------
@wp.kernel
def solve_floor_contacts(
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    x_prev: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.quat),
    inv_mass: wp.array(dtype=float),
    inv_I: wp.array(dtype=wp.vec3),
    he: wp.array(dtype=wp.vec3),
    shape_type: wp.array(dtype=int),
    floor_y: float,
    mu_s: float,
    lam_floor: wp.array(dtype=float),   # n_boxes * 8
    h: float,
    # outputs (Jacobi accumulators)
    dx: wp.array(dtype=wp.vec3),
    drot: wp.array(dtype=wp.vec3),
    dcount: wp.array(dtype=int),
):
    i = wp.tid()
    if inv_mass[i] == 0.0:
        return
    if shape_type[i] != 0:              # capsules/spheres → solve_floor_capsule
        return
    s = wp.vec3(0.0, 1.0, 0.0)  # floor separation normal points up
    for c in range(8):
        corner = box_corner(he[i], c)
        p1 = x[i] + wp.quat_rotate(q[i], corner)
        d = floor_y - p1[1]
        slot = i * 8 + c
        if d <= 0.0:
            lam_floor[slot] = 0.0
            continue
        r = p1 - x[i]
        w = gen_inv_mass(inv_mass[i], q[i], inv_I[i], r, s)
        if w <= 0.0:
            continue
        dlam = d / w                       # α = 0 contact
        lam_floor[slot] = dlam
        p = dlam * s
        # body A = box (+), B = static floor
        wp.atomic_add(dx, i, p * inv_mass[i])
        wp.atomic_add(drot, i, world_invI_mul(q[i], inv_I[i], wp.cross(r, p)))
        wp.atomic_add(dcount, i, 1)
        # position-level static friction (§3.5): cancel tangential slip if it
        # stays inside the cone λ_t < μ_s λ_n.
        if mu_s > 0.0:
            p1b = x_prev[i] + wp.quat_rotate(q_prev[i], corner)
            dp = (p1 - p1b)
            dpt = dp - s * wp.dot(dp, s)
            lt = wp.length(dpt)
            if lt > EPS and lt < mu_s * d:
                w2 = gen_inv_mass(inv_mass[i], q[i], inv_I[i], r, dpt / lt)
                if w2 > 0.0:
                    pf = -dpt / w2
                    wp.atomic_add(dx, i, pf * inv_mass[i])
                    wp.atomic_add(drot, i, world_invI_mul(q[i], inv_I[i], wp.cross(r, pf)))


@wp.kernel
def solve_floor_capsule(
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    x_prev: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.quat),
    inv_mass: wp.array(dtype=float),
    inv_I: wp.array(dtype=wp.vec3),
    shape_type: wp.array(dtype=int),
    shape_r: wp.array(dtype=float),
    shape_hl: wp.array(dtype=float),
    floor_y: float,
    mu_s: float,
    lam_floor: wp.array(dtype=float),
    h: float,
    dx: wp.array(dtype=wp.vec3),
    drot: wp.array(dtype=wp.vec3),
    dcount: wp.array(dtype=int),
):
    """Capsule/sphere vs floor: the two segment endpoints (one for a sphere) are
    tested as spheres against the plane y = floor_y. Mirrors the box-corner path
    (slots 0..1 of ``lam_floor``)."""
    i = wp.tid()
    if inv_mass[i] == 0.0:
        return
    if shape_type[i] != 1:             # capsule/sphere only (cylinder → its own kernel)
        return
    s = wp.vec3(0.0, 1.0, 0.0)
    rc = shape_r[i]
    hl = shape_hl[i]
    nend = wp.where(hl < EPS, 1, 2)
    for c in range(2):
        slot = i * 8 + c
        if c >= nend:
            lam_floor[slot] = 0.0
            continue
        sgn = wp.where(c == 0, -1.0, 1.0)
        off = wp.vec3(0.0, 0.0, sgn * hl)
        p1 = x[i] + wp.quat_rotate(q[i], off) - s * rc   # lowest cap-surface point
        d = floor_y - p1[1]
        if d <= 0.0:
            lam_floor[slot] = 0.0
            continue
        r = p1 - x[i]
        w = gen_inv_mass(inv_mass[i], q[i], inv_I[i], r, s)
        if w <= 0.0:
            continue
        dlam = d / w
        lam_floor[slot] = dlam
        p = dlam * s
        wp.atomic_add(dx, i, p * inv_mass[i])
        wp.atomic_add(drot, i, world_invI_mul(q[i], inv_I[i], wp.cross(r, p)))
        wp.atomic_add(dcount, i, 1)
        if mu_s > 0.0:
            p1b = x_prev[i] + wp.quat_rotate(q_prev[i], off) - s * rc
            dp = p1 - p1b
            dpt = dp - s * wp.dot(dp, s)
            lt = wp.length(dpt)
            if lt > EPS and lt < mu_s * d:
                w2 = gen_inv_mass(inv_mass[i], q[i], inv_I[i], r, dpt / lt)
                if w2 > 0.0:
                    pf = -dpt / w2
                    wp.atomic_add(dx, i, pf * inv_mass[i])
                    wp.atomic_add(drot, i, world_invI_mul(q[i], inv_I[i], wp.cross(r, pf)))


@wp.kernel
def solve_floor_cylinder(
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    x_prev: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.quat),
    inv_mass: wp.array(dtype=float),
    inv_I: wp.array(dtype=wp.vec3),
    shape_type: wp.array(dtype=int),
    shape_r: wp.array(dtype=float),
    shape_hl: wp.array(dtype=float),
    floor_y: float,
    mu_s: float,
    lam_floor: wp.array(dtype=float),
    h: float,
    dx: wp.array(dtype=wp.vec3),
    drot: wp.array(dtype=wp.vec3),
    dcount: wp.array(dtype=int),
):
    """Cylinder vs floor with **flat caps** (unlike a capsule). Each cap's rim is
    sampled at 4 points — one aligned with the downhill direction, so the true
    lowest rim point is always tested. This rests a vertical cylinder on its flat
    face, a horizontal one on a side line, a tilted one on a rim edge — exactly a
    cylinder, not a rounded capsule. Uses all 8 ``lam_floor`` slots (2 caps × 4)."""
    i = wp.tid()
    if inv_mass[i] == 0.0:
        return
    if shape_type[i] != 2:
        return
    s = wp.vec3(0.0, 1.0, 0.0)
    r = shape_r[i]
    hl = shape_hl[i]
    a = wp.quat_rotate(q[i], wp.vec3(0.0, 0.0, 1.0))    # cylinder axis (world)
    perp = s - a * wp.dot(s, a)                         # "up" projected ⟂ to axis
    plen = wp.length(perp)
    if plen > 1.0e-5:
        d1 = -perp / plen                              # downhill in the rim plane
    else:                                              # axis ≈ vertical → any basis
        tmp = wp.vec3(1.0, 0.0, 0.0)
        if wp.abs(a[0]) > 0.9:
            tmp = wp.vec3(0.0, 1.0, 0.0)
        d1 = wp.normalize(wp.cross(a, tmp))
    d2 = wp.normalize(wp.cross(a, d1))
    for cap in range(2):
        capoff = a * (wp.where(cap == 0, -1.0, 1.0) * hl)
        for jj in range(4):
            if jj == 0:
                dirv = d1
            elif jj == 1:
                dirv = d2
            elif jj == 2:
                dirv = d1 * (-1.0)
            else:
                dirv = d2 * (-1.0)
            roff = capoff + dirv * r                    # rim point, offset from centre
            p1 = x[i] + roff
            slot = i * 8 + cap * 4 + jj
            d = floor_y - p1[1]
            if d <= 0.0:
                lam_floor[slot] = 0.0
                continue
            w = gen_inv_mass(inv_mass[i], q[i], inv_I[i], roff, s)
            if w <= 0.0:
                continue
            dlam = d / w
            lam_floor[slot] = dlam
            p = dlam * s
            wp.atomic_add(dx, i, p * inv_mass[i])
            wp.atomic_add(drot, i, world_invI_mul(q[i], inv_I[i], wp.cross(roff, p)))
            wp.atomic_add(dcount, i, 1)
            if mu_s > 0.0:
                roff_local = wp.quat_rotate_inv(q[i], roff)
                p1b = x_prev[i] + wp.quat_rotate(q_prev[i], roff_local)
                dp = p1 - p1b
                dpt = dp - s * wp.dot(dp, s)
                lt = wp.length(dpt)
                if lt > EPS and lt < mu_s * d:
                    w2 = gen_inv_mass(inv_mass[i], q[i], inv_I[i], roff, dpt / lt)
                    if w2 > 0.0:
                        pf = -dpt / w2
                        wp.atomic_add(dx, i, pf * inv_mass[i])
                        wp.atomic_add(drot, i, world_invI_mul(q[i], inv_I[i], wp.cross(roff, pf)))


# -----------------------------------------------------------------------------
# Box-box contact manifold: face-axis SAT + Sutherland-Hodgman face clip
# -----------------------------------------------------------------------------
# Corner-vs-OBB fails for aligned face-face stacking (the upper box's corners
# sit exactly on the lower box's edges, never strictly inside). The standard fix
# is SAT to pick a single contact normal (the min-overlap face axis), then clip
# the incident face against the reference face to get up to 8 manifold points.
# We generate the manifold ONCE per substep (normal frozen, à la VBD §3.5), then
# the position + velocity passes read it. MAXC = max manifold points per pair.
MAXC = wp.constant(8)


@wp.func
def proj_radius(L: wp.vec3, R: wp.mat33, e: wp.vec3) -> float:
    return (wp.abs(e[0] * wp.dot(L, wp.vec3(R[0, 0], R[1, 0], R[2, 0])))
            + wp.abs(e[1] * wp.dot(L, wp.vec3(R[0, 1], R[1, 1], R[2, 1])))
            + wp.abs(e[2] * wp.dot(L, wp.vec3(R[0, 2], R[1, 2], R[2, 2]))))


@wp.kernel
def generate_box_manifold(
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    he: wp.array(dtype=wp.vec3),
    shape_type: wp.array(dtype=int),
    pair_a: wp.array(dtype=int),
    pair_b: wp.array(dtype=int),
    pair_count: wp.array(dtype=int),     # device-side live pair count (graph gate)
    margin: float,
    poly: wp.array(dtype=wp.vec3),       # scratch: 16 vec3 per pair
    # outputs
    m_count: wp.array(dtype=int),
    m_inc: wp.array(dtype=int),
    m_ref: wp.array(dtype=int),
    m_normal: wp.array(dtype=wp.vec3),   # ref-face outward normal (inc pushed +n)
    m_off_inc: wp.array(dtype=wp.vec3),  # per pair × MAXC
    m_off_ref: wp.array(dtype=wp.vec3),
):
    pid = wp.tid()
    m_count[pid] = 0
    if pid >= pair_count[0]:             # gated: launch dim is fixed capacity
        return
    a = pair_a[pid]
    b = pair_b[pid]
    if shape_type[a] != 0 or shape_type[b] != 0:
        return                           # non-box pair → handled by capsule kernel
    cA = x[a]; cB = x[b]
    eA = he[a]; eB = he[b]
    RA = wp.quat_to_matrix(q[a])
    RB = wp.quat_to_matrix(q[b])
    cBA = cA - cB                        # from B to A

    # --- face-axis SAT (6 axes: 3 of A, 3 of B) ---
    best_overlap = float(1.0e30)
    best_axis = wp.vec3(0.0, 1.0, 0.0)
    ref_is_a = int(1)
    best_k = int(0)
    for k in range(3):
        L = wp.vec3(RA[0, k], RA[1, k], RA[2, k])
        ov = eA[k] + proj_radius(L, RB, eB) - wp.abs(wp.dot(L, cBA))
        if ov < 0.0:
            return
        if ov < best_overlap:
            best_overlap = ov; best_axis = L; ref_is_a = 1; best_k = k
    for k in range(3):
        L = wp.vec3(RB[0, k], RB[1, k], RB[2, k])
        ov = eB[k] + proj_radius(L, RA, eA) - wp.abs(wp.dot(L, cBA))
        if ov < 0.0:
            return
        if ov < best_overlap:
            best_overlap = ov; best_axis = L; ref_is_a = 0; best_k = k

    # nBA points B→A; ref-face outward normal points ref→inc.
    nBA = best_axis
    if wp.dot(nBA, cBA) < 0.0:
        nBA = -nBA
    if ref_is_a == 1:
        ref_b = a; inc_b = b
        c_ref = cA; R_ref = RA; e_ref = eA
        c_inc = cB; R_inc = RB; e_inc = eB
        ref_face_n = -nBA
    else:
        ref_b = b; inc_b = a
        c_ref = cB; R_ref = RB; e_ref = eB
        c_inc = cA; R_inc = RA; e_inc = eA
        ref_face_n = nBA

    ref_axis = best_k
    ref_face_c = c_ref + ref_face_n * e_ref[ref_axis]
    ax1 = (ref_axis + 1) % 3
    ax2 = (ref_axis + 2) % 3
    u_ref = wp.vec3(R_ref[0, ax1], R_ref[1, ax1], R_ref[2, ax1])
    v_ref = wp.vec3(R_ref[0, ax2], R_ref[1, ax2], R_ref[2, ax2])
    eu = e_ref[ax1]; ev = e_ref[ax2]

    # incident face = inc body's face most anti-parallel to ref_face_n
    inc_axis = int(0)
    inc_sign = float(1.0)
    best_dot = float(-1.0e30)
    for k in range(3):
        col = wp.vec3(R_inc[0, k], R_inc[1, k], R_inc[2, k])
        dp = -wp.dot(col, ref_face_n)
        if dp > best_dot:
            best_dot = dp; inc_axis = k; inc_sign = 1.0
        if -dp > best_dot:
            best_dot = -dp; inc_axis = k; inc_sign = -1.0
    inc_n = wp.vec3(R_inc[0, inc_axis], R_inc[1, inc_axis], R_inc[2, inc_axis]) * inc_sign
    inc_face_c = c_inc + inc_n * e_inc[inc_axis]
    iax1 = (inc_axis + 1) % 3
    iax2 = (inc_axis + 2) % 3
    iu = wp.vec3(R_inc[0, iax1], R_inc[1, iax1], R_inc[2, iax1])
    iv = wp.vec3(R_inc[0, iax2], R_inc[1, iax2], R_inc[2, iax2])
    ieu = e_inc[iax1]; iev = e_inc[iax2]

    cur = pid * 16
    nxt = pid * 16 + 8
    poly[cur + 0] = inc_face_c + iu * ieu + iv * iev
    poly[cur + 1] = inc_face_c - iu * ieu + iv * iev
    poly[cur + 2] = inc_face_c - iu * ieu - iv * iev
    poly[cur + 3] = inc_face_c + iu * ieu - iv * iev
    plen = int(4)
    for pi in range(4):
        if pi == 0:
            pp = ref_face_c + u_ref * eu; pn = u_ref
        elif pi == 1:
            pp = ref_face_c - u_ref * eu; pn = -u_ref
        elif pi == 2:
            pp = ref_face_c + v_ref * ev; pn = v_ref
        else:
            pp = ref_face_c - v_ref * ev; pn = -v_ref
        if plen == 0:
            return
        olen = int(0)
        for iv2 in range(plen):
            ap = poly[cur + iv2]
            bp = poly[cur + (iv2 + 1) % plen]
            da = wp.dot(ap - pp, pn)
            db = wp.dot(bp - pp, pn)
            if da <= 0.0:
                if olen < 8:
                    poly[nxt + olen] = ap; olen += 1
                if db > 0.0:
                    t = da / (da - db)
                    if olen < 8:
                        poly[nxt + olen] = ap + (bp - ap) * t; olen += 1
            else:
                if db <= 0.0:
                    t = da / (da - db)
                    if olen < 8:
                        poly[nxt + olen] = ap + (bp - ap) * t; olen += 1
        tmp = cur; cur = nxt; nxt = tmp
        plen = olen
    if plen == 0:
        return

    Rt_ref = wp.transpose(R_ref)
    Rt_inc = wp.transpose(R_inc)
    nc = int(0)
    for iv3 in range(plen):
        if nc >= 8:
            break
        p_inc = poly[cur + iv3]
        d_signed = wp.dot(p_inc - ref_face_c, ref_face_n)
        if d_signed < margin:                 # penetrating (or within warm margin)
            p_ref = p_inc - ref_face_n * d_signed
            m_off_inc[pid * 8 + nc] = Rt_inc * (p_inc - c_inc)
            m_off_ref[pid * 8 + nc] = Rt_ref * (p_ref - c_ref)
            nc += 1
    m_count[pid] = nc
    m_inc[pid] = inc_b
    m_ref[pid] = ref_b
    m_normal[pid] = ref_face_n


@wp.kernel
def solve_box_manifold(
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    inv_mass: wp.array(dtype=float),
    inv_I: wp.array(dtype=wp.vec3),
    m_count: wp.array(dtype=int),
    m_inc: wp.array(dtype=int),
    m_ref: wp.array(dtype=int),
    m_normal: wp.array(dtype=wp.vec3),
    m_off_inc: wp.array(dtype=wp.vec3),
    m_off_ref: wp.array(dtype=wp.vec3),
    lam_pair: wp.array(dtype=float),     # n_pairs * 8
    dx: wp.array(dtype=wp.vec3),
    drot: wp.array(dtype=wp.vec3),
    dcount: wp.array(dtype=int),
):
    pid = wp.tid()
    nc = m_count[pid]
    if nc == 0:
        return
    inc = m_inc[pid]
    ref = m_ref[pid]
    s = m_normal[pid]                    # push inc body along +s
    for c in range(nc):
        slot = pid * 8 + c
        p_inc = x[inc] + wp.quat_rotate(q[inc], m_off_inc[slot])
        p_ref = x[ref] + wp.quat_rotate(q[ref], m_off_ref[slot])
        C = wp.dot(p_inc - p_ref, s)      # <0 when penetrating
        if C >= 0.0:
            lam_pair[slot] = 0.0
            continue
        d = -C
        r_inc = p_inc - x[inc]
        r_ref = p_ref - x[ref]
        w_inc = gen_inv_mass(inv_mass[inc], q[inc], inv_I[inc], r_inc, s)
        w_ref = gen_inv_mass(inv_mass[ref], q[ref], inv_I[ref], r_ref, s)
        wsum = w_inc + w_ref
        if wsum <= 0.0:
            continue
        dlam = d / wsum
        lam_pair[slot] = lam_pair[slot] + dlam
        p = dlam * s
        accumulate_correction(dx, drot, dcount, inc, ref,
                              inv_mass[inc], inv_mass[ref], q[inc], q[ref],
                              inv_I[inc], inv_I[ref], r_inc, r_ref, p)


# -----------------------------------------------------------------------------
# Capsule / sphere narrow phase. A capsule is a segment (along body-local +Z,
# half-length ``hl``) inflated by radius ``r``; a sphere is ``hl = 0``. Contacts
# reduce to the closest distance between two such segments (or a segment and an
# OBB), then a sphere test — far simpler than the box SAT + clip above, and they
# fill the SAME manifold buffers, so ``solve_box_manifold`` / ``velocity_box``
# resolve them unchanged (the contact solve is geometry-agnostic).
# -----------------------------------------------------------------------------
@wp.func
def capsule_seg(xc: wp.vec3, qc: wp.quat, hl: float):
    """World endpoints of a capsule's core segment (body-local +Z axis)."""
    ax = wp.quat_rotate(qc, wp.vec3(0.0, 0.0, 1.0))
    return xc - ax * hl, xc + ax * hl


@wp.func
def closest_seg_seg(p1: wp.vec3, q1: wp.vec3, p2: wp.vec3, q2: wp.vec3):
    """Closest points between segments [p1,q1] and [p2,q2] (Ericson §5.1.9)."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = wp.dot(d1, d1)
    e = wp.dot(d2, d2)
    f = wp.dot(d2, r)
    s = float(0.0)
    t = float(0.0)
    if a <= EPS and e <= EPS:
        return p1, p2
    if a <= EPS:
        t = wp.clamp(f / e, 0.0, 1.0)
    else:
        c = wp.dot(d1, r)
        if e <= EPS:
            s = wp.clamp(-c / a, 0.0, 1.0)
        else:
            b = wp.dot(d1, d2)
            denom = a * e - b * b
            if denom > EPS:
                s = wp.clamp((b * f - c * e) / denom, 0.0, 1.0)
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = wp.clamp(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = wp.clamp((b - c) / a, 0.0, 1.0)
    return p1 + d1 * s, p2 + d2 * t


@wp.func
def closest_pt_obb(p: wp.vec3, c: wp.vec3, qb: wp.quat, he: wp.vec3) -> wp.vec3:
    """Closest point on box (c, qb, he) to world point ``p``."""
    local = wp.quat_rotate_inv(qb, p - c)
    cl = wp.vec3(wp.clamp(local[0], -he[0], he[0]),
                 wp.clamp(local[1], -he[1], he[1]),
                 wp.clamp(local[2], -he[2], he[2]))
    return c + wp.quat_rotate(qb, cl)


@wp.kernel
def generate_capsule_manifold(
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    he: wp.array(dtype=wp.vec3),
    shape_type: wp.array(dtype=int),
    shape_r: wp.array(dtype=float),
    shape_hl: wp.array(dtype=float),
    pair_a: wp.array(dtype=int),
    pair_b: wp.array(dtype=int),
    pair_count: wp.array(dtype=int),
    margin: float,
    # outputs (same buffers as the box manifold)
    m_count: wp.array(dtype=int),
    m_inc: wp.array(dtype=int),
    m_ref: wp.array(dtype=int),
    m_normal: wp.array(dtype=wp.vec3),
    m_off_inc: wp.array(dtype=wp.vec3),
    m_off_ref: wp.array(dtype=wp.vec3),
):
    pid = wp.tid()
    if pid >= pair_count[0]:
        return
    a = pair_a[pid]
    b = pair_b[pid]
    sa = shape_type[a]
    sb = shape_type[b]
    if sa == 0 and sb == 0:
        return                              # box-box → handled by box manifold

    if sa != 0 and sb != 0:
        # capsule/sphere/cylinder vs same: treat each as a segment + radius (a
        # cylinder uses its capsule approximation here for body-body contacts; its
        # floor contact is handled exactly by solve_floor_cylinder). 1 contact.
        p1, e1 = capsule_seg(x[a], q[a], shape_hl[a])
        p2, e2 = capsule_seg(x[b], q[b], shape_hl[b])
        c1, c2 = closest_seg_seg(p1, e1, p2, e2)
        delta = c1 - c2
        dist = wp.length(delta)
        ra = shape_r[a]
        rb = shape_r[b]
        if dist - (ra + rb) < margin:
            n = wp.vec3(0.0, 1.0, 0.0)
            if dist > EPS:
                n = delta / dist            # B→A; inc=A pushed +n
            pa = c1 - n * ra
            pb = c2 + n * rb
            m_off_inc[pid * 8 + 0] = wp.quat_rotate_inv(q[a], pa - x[a])
            m_off_ref[pid * 8 + 0] = wp.quat_rotate_inv(q[b], pb - x[b])
            m_normal[pid] = n
            m_inc[pid] = a
            m_ref[pid] = b
            m_count[pid] = 1
        else:
            m_count[pid] = 0
        return

    # box vs capsule: box is the reference, capsule the incident. Use the capsule
    # endpoint nearest the box (sphere-vs-OBB) as a single contact.
    boxb = wp.where(sa == 0, a, b)
    capb = wp.where(sa == 0, b, a)
    rc = shape_r[capb]
    pc0, pc1 = capsule_seg(x[capb], q[capb], shape_hl[capb])
    cp0 = closest_pt_obb(pc0, x[boxb], q[boxb], he[boxb])
    cp1 = closest_pt_obb(pc1, x[boxb], q[boxb], he[boxb])
    d0 = wp.length(pc0 - cp0)
    d1 = wp.length(pc1 - cp1)
    # pick the endpoint with the smaller surface separation (deepest contact)
    ec = pc0
    cb = cp0
    dd = d0
    if (d1 - rc) < (d0 - rc):
        ec = pc1
        cb = cp1
        dd = d1
    if dd - rc < margin:
        inside, sout, dpen = obb_penetration(ec, x[boxb], q[boxb], he[boxb])
        n = sout                            # box→capsule
        pref = cb
        if not inside and dd > EPS:
            n = (ec - cb) / dd
            pref = cb
        else:
            pref = ec - n * dpen            # project centre to nearest box face
        pinc = ec - n * rc
        m_off_inc[pid * 8 + 0] = wp.quat_rotate_inv(q[capb], pinc - x[capb])
        m_off_ref[pid * 8 + 0] = wp.quat_rotate_inv(q[boxb], pref - x[boxb])
        m_normal[pid] = n
        m_inc[pid] = capb
        m_ref[pid] = boxb
        m_count[pid] = 1
    else:
        m_count[pid] = 0


# -----------------------------------------------------------------------------
# Joints: compliant XPBD distance/attachment constraint between local anchor
# points on any two bodies (Macklin 2016 Eq. 18; Müller 2020 §3.3 positional
# constraint). Covers cloth edges (anchors at body centres, particles with
# inv_I=0), rigid-rigid links and pin-to-rigid attachments (rest=0). Fixed
# topology → solved inside the captured graph alongside contacts.
# -----------------------------------------------------------------------------
@wp.kernel
def solve_joints(
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    inv_mass: wp.array(dtype=float),
    inv_I: wp.array(dtype=wp.vec3),
    j_a: wp.array(dtype=int),
    j_b: wp.array(dtype=int),
    j_anchor_a: wp.array(dtype=wp.vec3),   # anchor in A's body frame
    j_anchor_b: wp.array(dtype=wp.vec3),   # anchor in B's body frame
    j_rest: wp.array(dtype=float),
    j_compliance: wp.array(dtype=float),
    lam_joint: wp.array(dtype=float),      # one λ per joint (reset per substep)
    h: float,
    dx: wp.array(dtype=wp.vec3),
    drot: wp.array(dtype=wp.vec3),
    dcount: wp.array(dtype=int),
):
    k = wp.tid()
    a = j_a[k]
    b = j_b[k]
    pa = x[a] + wp.quat_rotate(q[a], j_anchor_a[k])
    pb = x[b] + wp.quat_rotate(q[b], j_anchor_b[k])
    d = pa - pb
    dist = wp.length(d)
    if dist < EPS:
        return
    n = d / dist
    C = dist - j_rest[k]                   # >0 stretched, <0 compressed
    ra = pa - x[a]
    rb = pb - x[b]
    w_a = gen_inv_mass(inv_mass[a], q[a], inv_I[a], ra, n)
    w_b = gen_inv_mass(inv_mass[b], q[b], inv_I[b], rb, n)
    wsum = w_a + w_b
    if wsum <= 0.0:
        return
    # compliant XPBD update: Δλ = (−C − α̃ λ)/(w + α̃), α̃ = compliance/h²
    alpha = j_compliance[k] / (h * h)
    dlam = (-C - alpha * lam_joint[k]) / (wsum + alpha)
    lam_joint[k] = lam_joint[k] + dlam
    p = dlam * n
    accumulate_correction(dx, drot, dcount, a, b,
                          inv_mass[a], inv_mass[b], q[a], q[b],
                          inv_I[a], inv_I[b], ra, rb, p)


@wp.kernel
def apply_jacobi_6dof(
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    inv_mass: wp.array(dtype=float),
    relax: float,
    dx: wp.array(dtype=wp.vec3),
    drot: wp.array(dtype=wp.vec3),
    dcount: wp.array(dtype=int),
):
    i = wp.tid()
    n = dcount[i]
    if n > 0 and inv_mass[i] > 0.0:
        inv_n = relax / float(n)
        x[i] = x[i] + dx[i] * inv_n
        dw = drot[i] * inv_n
        wq = wp.quat(dw[0], dw[1], dw[2], 0.0)
        q[i] = wp.normalize(q[i] + 0.5 * (wq * q[i]))
    dx[i] = wp.vec3(0.0, 0.0, 0.0)
    drot[i] = wp.vec3(0.0, 0.0, 0.0)
    dcount[i] = 0


# -----------------------------------------------------------------------------
# Velocity finalize (Algorithm 2: v = (x − x_prev)/h; ω from Δq) + damping
# -----------------------------------------------------------------------------
@wp.kernel
def finalize_velocity_6dof(
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    x_prev: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.quat),
    inv_mass: wp.array(dtype=float),
    h: float,
    lin_damp: float,
    ang_damp: float,
    v: wp.array(dtype=wp.vec3),
    omega: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    if inv_mass[i] == 0.0:
        v[i] = wp.vec3(0.0, 0.0, 0.0)
        omega[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    v[i] = (x[i] - x_prev[i]) / h * (1.0 - lin_damp)
    dq = q[i] * wp.quat_inverse(q_prev[i])
    w = wp.vec3(dq[0], dq[1], dq[2]) * (2.0 / h)
    if dq[3] < 0.0:
        w = -w
    omega[i] = w * (1.0 - ang_damp)


# -----------------------------------------------------------------------------
# Velocity-level dynamic friction + restitution (Algorithm 2: SolveVelocities)
# -----------------------------------------------------------------------------
@wp.kernel
def velocity_floor(
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    inv_mass: wp.array(dtype=float),
    inv_I: wp.array(dtype=wp.vec3),
    he: wp.array(dtype=wp.vec3),
    shape_type: wp.array(dtype=int),
    floor_y: float,
    mu_d: float,
    restitution: float,
    lam_floor: wp.array(dtype=float),
    h: float,
    v: wp.array(dtype=wp.vec3),
    omega: wp.array(dtype=wp.vec3),
    dv: wp.array(dtype=wp.vec3),
    dw: wp.array(dtype=wp.vec3),
    dvc: wp.array(dtype=int),
):
    i = wp.tid()
    if inv_mass[i] == 0.0:
        return
    if shape_type[i] != 0:              # capsules/spheres → velocity_floor_capsule
        return
    s = wp.vec3(0.0, 1.0, 0.0)
    for c in range(8):
        slot = i * 8 + c
        lam_n = lam_floor[slot]
        if lam_n == 0.0:
            continue
        corner = box_corner(he[i], c)
        r = wp.quat_rotate(q[i], corner)
        vc = v[i] + wp.cross(omega[i], r)        # contact-point velocity
        vn = wp.dot(vc, s)
        vt = vc - s * vn
        ltan = wp.length(vt)
        impulse = wp.vec3(0.0, 0.0, 0.0)
        # dynamic friction (Eq. 30): Δv = −v_t/|v_t| · min(h μ_d |f_n|, |v_t|)
        if mu_d > 0.0 and ltan > EPS:
            fn = lam_n / (h * h)
            dvm = wp.min(h * mu_d * wp.abs(fn), ltan)
            impulse = impulse - (vt / ltan) * dvm
        # restitution (Eq. for normal velocity): remove inward normal velocity
        # and add e·(−v_n) outward. Threshold avoids resting jitter.
        if vn < 0.0:
            impulse = impulse + s * (-vn + wp.max(-restitution * vn, 0.0))
        if wp.length(impulse) > EPS:
            w = gen_inv_mass(inv_mass[i], q[i], inv_I[i], r, impulse / wp.length(impulse))
            if w > 0.0:
                p = impulse / w
                wp.atomic_add(dv, i, p * inv_mass[i])
                wp.atomic_add(dw, i, world_invI_mul(q[i], inv_I[i], wp.cross(r, p)))
                wp.atomic_add(dvc, i, 1)


@wp.kernel
def velocity_floor_capsule(
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    inv_mass: wp.array(dtype=float),
    inv_I: wp.array(dtype=wp.vec3),
    shape_type: wp.array(dtype=int),
    shape_r: wp.array(dtype=float),
    shape_hl: wp.array(dtype=float),
    floor_y: float,
    mu_d: float,
    restitution: float,
    lam_floor: wp.array(dtype=float),
    h: float,
    v: wp.array(dtype=wp.vec3),
    omega: wp.array(dtype=wp.vec3),
    dv: wp.array(dtype=wp.vec3),
    dw: wp.array(dtype=wp.vec3),
    dvc: wp.array(dtype=int),
):
    """Dynamic friction + restitution for capsule/sphere floor contacts."""
    i = wp.tid()
    if inv_mass[i] == 0.0:
        return
    if shape_type[i] != 1:             # capsule/sphere only (cylinder → its own kernel)
        return
    s = wp.vec3(0.0, 1.0, 0.0)
    rc = shape_r[i]
    hl = shape_hl[i]
    for c in range(2):
        slot = i * 8 + c
        lam_n = lam_floor[slot]
        if lam_n == 0.0:
            continue
        sgn = wp.where(c == 0, -1.0, 1.0)
        off = wp.vec3(0.0, 0.0, sgn * hl)
        r = wp.quat_rotate(q[i], off) - s * rc
        vc = v[i] + wp.cross(omega[i], r)
        vn = wp.dot(vc, s)
        vt = vc - s * vn
        ltan = wp.length(vt)
        impulse = wp.vec3(0.0, 0.0, 0.0)
        if mu_d > 0.0 and ltan > EPS:
            fn = lam_n / (h * h)
            dvm = wp.min(h * mu_d * wp.abs(fn), ltan)
            impulse = impulse - (vt / ltan) * dvm
        if vn < 0.0:
            impulse = impulse + s * (-vn + wp.max(-restitution * vn, 0.0))
        if wp.length(impulse) > EPS:
            w = gen_inv_mass(inv_mass[i], q[i], inv_I[i], r, impulse / wp.length(impulse))
            if w > 0.0:
                p = impulse / w
                wp.atomic_add(dv, i, p * inv_mass[i])
                wp.atomic_add(dw, i, world_invI_mul(q[i], inv_I[i], wp.cross(r, p)))
                wp.atomic_add(dvc, i, 1)


@wp.kernel
def velocity_floor_cylinder(
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    inv_mass: wp.array(dtype=float),
    inv_I: wp.array(dtype=wp.vec3),
    shape_type: wp.array(dtype=int),
    shape_r: wp.array(dtype=float),
    shape_hl: wp.array(dtype=float),
    floor_y: float,
    mu_d: float,
    restitution: float,
    lam_floor: wp.array(dtype=float),
    h: float,
    v: wp.array(dtype=wp.vec3),
    omega: wp.array(dtype=wp.vec3),
    dv: wp.array(dtype=wp.vec3),
    dw: wp.array(dtype=wp.vec3),
    dvc: wp.array(dtype=int),
):
    """Dynamic friction + restitution for cylinder floor contacts (same 8 rim
    samples as ``solve_floor_cylinder``)."""
    i = wp.tid()
    if inv_mass[i] == 0.0:
        return
    if shape_type[i] != 2:
        return
    s = wp.vec3(0.0, 1.0, 0.0)
    r = shape_r[i]
    hl = shape_hl[i]
    a = wp.quat_rotate(q[i], wp.vec3(0.0, 0.0, 1.0))
    perp = s - a * wp.dot(s, a)
    plen = wp.length(perp)
    if plen > 1.0e-5:
        d1 = -perp / plen
    else:
        tmp = wp.vec3(1.0, 0.0, 0.0)
        if wp.abs(a[0]) > 0.9:
            tmp = wp.vec3(0.0, 1.0, 0.0)
        d1 = wp.normalize(wp.cross(a, tmp))
    d2 = wp.normalize(wp.cross(a, d1))
    for cap in range(2):
        capoff = a * (wp.where(cap == 0, -1.0, 1.0) * hl)
        for jj in range(4):
            slot = i * 8 + cap * 4 + jj
            lam_n = lam_floor[slot]
            if lam_n == 0.0:
                continue
            if jj == 0:
                dirv = d1
            elif jj == 1:
                dirv = d2
            elif jj == 2:
                dirv = d1 * (-1.0)
            else:
                dirv = d2 * (-1.0)
            roff = capoff + dirv * r
            vc = v[i] + wp.cross(omega[i], roff)
            vn = wp.dot(vc, s)
            vt = vc - s * vn
            ltan = wp.length(vt)
            impulse = wp.vec3(0.0, 0.0, 0.0)
            if mu_d > 0.0 and ltan > EPS:
                fn = lam_n / (h * h)
                dvm = wp.min(h * mu_d * wp.abs(fn), ltan)
                impulse = impulse - (vt / ltan) * dvm
            if vn < 0.0:
                impulse = impulse + s * (-vn + wp.max(-restitution * vn, 0.0))
            if wp.length(impulse) > EPS:
                w = gen_inv_mass(inv_mass[i], q[i], inv_I[i], roff, impulse / wp.length(impulse))
                if w > 0.0:
                    p = impulse / w
                    wp.atomic_add(dv, i, p * inv_mass[i])
                    wp.atomic_add(dw, i, world_invI_mul(q[i], inv_I[i], wp.cross(roff, p)))
                    wp.atomic_add(dvc, i, 1)


@wp.kernel
def velocity_box(
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    inv_mass: wp.array(dtype=float),
    inv_I: wp.array(dtype=wp.vec3),
    m_count: wp.array(dtype=int),
    m_inc: wp.array(dtype=int),
    m_ref: wp.array(dtype=int),
    m_normal: wp.array(dtype=wp.vec3),
    m_off_inc: wp.array(dtype=wp.vec3),
    m_off_ref: wp.array(dtype=wp.vec3),
    mu_d: float,
    restitution: float,
    lam_pair: wp.array(dtype=float),
    h: float,
    v: wp.array(dtype=wp.vec3),
    omega: wp.array(dtype=wp.vec3),
    dv: wp.array(dtype=wp.vec3),
    dw: wp.array(dtype=wp.vec3),
    dvc: wp.array(dtype=int),
):
    pid = wp.tid()
    nc = m_count[pid]
    if nc == 0:
        return
    ci = m_inc[pid]
    cj = m_ref[pid]
    s = m_normal[pid]
    for c in range(nc):
        slot = pid * 8 + c
        lam_n = lam_pair[slot]
        if lam_n == 0.0:
            continue
        p_inc = x[ci] + wp.quat_rotate(q[ci], m_off_inc[slot])
        p_ref = x[cj] + wp.quat_rotate(q[cj], m_off_ref[slot])
        ri = p_inc - x[ci]
        rj = p_ref - x[cj]
        vci = v[ci] + wp.cross(omega[ci], ri)
        vcj = v[cj] + wp.cross(omega[cj], rj)
        vrel = vci - vcj
        vn = wp.dot(vrel, s)
        vt = vrel - s * vn
        ltan = wp.length(vt)
        impulse = wp.vec3(0.0, 0.0, 0.0)
        if mu_d > 0.0 and ltan > EPS:
            fn = lam_n / (h * h)
            dvm = wp.min(h * mu_d * wp.abs(fn), ltan)
            impulse = impulse - (vt / ltan) * dvm
        if vn < 0.0:
            impulse = impulse + s * (-vn + wp.max(-restitution * vn, 0.0))
        li = wp.length(impulse)
        if li > EPS:
            ndir = impulse / li
            wi = gen_inv_mass(inv_mass[ci], q[ci], inv_I[ci], ri, ndir)
            wj = gen_inv_mass(inv_mass[cj], q[cj], inv_I[cj], rj, ndir)
            wsum = wi + wj
            if wsum > 0.0:
                p = impulse / wsum
                if inv_mass[ci] > 0.0:
                    wp.atomic_add(dv, ci, p * inv_mass[ci])
                    wp.atomic_add(dw, ci, world_invI_mul(q[ci], inv_I[ci], wp.cross(ri, p)))
                    wp.atomic_add(dvc, ci, 1)
                if inv_mass[cj] > 0.0:
                    wp.atomic_add(dv, cj, -p * inv_mass[cj])
                    wp.atomic_add(dw, cj, -world_invI_mul(q[cj], inv_I[cj], wp.cross(rj, p)))
                    wp.atomic_add(dvc, cj, 1)


@wp.kernel
def apply_velocity_6dof(
    inv_mass: wp.array(dtype=float),
    dv: wp.array(dtype=wp.vec3),
    dw: wp.array(dtype=wp.vec3),
    dvc: wp.array(dtype=int),
    v: wp.array(dtype=wp.vec3),
    omega: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    n = dvc[i]
    if n > 0 and inv_mass[i] > 0.0:
        inv_n = 1.0 / float(n)
        v[i] = v[i] + dv[i] * inv_n
        omega[i] = omega[i] + dw[i] * inv_n
    dv[i] = wp.vec3(0.0, 0.0, 0.0)
    dw[i] = wp.vec3(0.0, 0.0, 0.0)
    dvc[i] = 0


@wp.kernel
def cap_velocity_6dof(v: wp.array(dtype=wp.vec3), omega: wp.array(dtype=wp.vec3),
                      max_lin: float, max_ang: float):
    i = wp.tid()
    sp = wp.length(v[i])
    if max_lin > 0.0 and sp > max_lin:
        v[i] = v[i] * (max_lin / sp)
    sa = wp.length(omega[i])
    if max_ang > 0.0 and sa > max_ang:
        omega[i] = omega[i] * (max_ang / sa)
