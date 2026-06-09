"""Warp kernels for the XPBD 3D particle solver.

Equation references are to the XPBD paper unless noted:

    Macklin, Müller, Chentanez. "XPBD: Position-Based Simulation of Compliant
    Constrained Dynamics." Motion in Games (MiG) 2016.   reference/XPBD_Macklin2016.pdf

The whole solver is a faithful transcription of the paper's Algorithm 1 with
small-substep integration (Müller et al. 2020, "Detailed Rigid Body Simulation
with Extended Position-Based Dynamics") layered on top: each frame is split into
``substeps`` integration steps, each running ``iterations`` constraint sweeps,
and ``λ`` is re-initialised to 0 at the start of every substep (Alg. 1 line 4).

Per-constraint compliance ``α`` (inverse stiffness, units m/N) is the paper's
headline contribution. The time-step-scaled compliance is

    α̃ = α / Δt²                                                     (§4)

and the single-constraint Gauss-Seidel multiplier update is

    Δλ_j = (−C_j − α̃_j λ_j) / (∇C_j Mⁱ ∇C_jᵀ + α̃_j)               (Eq. 18)
    Δx   = Mⁱ ∇C_jᵀ Δλ_j                                            (Eq. 17)

A compliance of 0 recovers an infinitely stiff (hard) PBD constraint. Contacts
are always hard (α = 0); the paper assumes zero compliance in contact (§6).

Constraint type codes (must match solver.py):
    0 DISTANCE : ‖x_a − x_b‖ − rest = 0          (springs, cloth, chain links)
    1 ATTACH   : ‖x_a − anchor‖ − rest = 0       (compliant pin to a world point)
    2 FLOOR    : one-sided x_a·ŷ ≥ floor_y       (push-only + Coulomb friction)
    3 CONTACT  : one-sided ‖x_a − x_b‖ ≥ rest    (sphere-sphere + friction)
"""

import warp as wp

# ---- Constraint type codes --------------------------------------------------
DISTANCE = wp.constant(0)
ATTACH = wp.constant(1)
FLOOR = wp.constant(2)
CONTACT = wp.constant(3)

EPS = wp.constant(1.0e-9)


# -----------------------------------------------------------------------------
# Integration / prediction  (Algorithm 1, line 1)
# -----------------------------------------------------------------------------
@wp.kernel
def integrate(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    gravity: wp.vec3,
    dt: float,
    # outputs
    x_prev: wp.array(dtype=wp.vec3),
):
    """Symplectic-Euler predictor for one substep:

        x_prev ← xⁿ
        v      ← vⁿ + Δt g                (external force = gravity, Mⁱ folded in)
        x      ← xⁿ + Δt v                (= x̃, the predicted/inertial position)

    Static bodies (``inv_mass == 0``) keep their position; their velocity is
    irrelevant. ``x_prev`` is read back by ``finalize_velocity`` (Alg. 1 line 16)
    and by the contact kernels for position-based friction.
    """
    i = wp.tid()
    x_prev[i] = x[i]
    if inv_mass[i] == 0.0:
        return
    v[i] = v[i] + gravity * dt
    x[i] = x[i] + v[i] * dt


# -----------------------------------------------------------------------------
# Position-based Coulomb friction helper (Müller et al. 2020, §3.5)
# -----------------------------------------------------------------------------
@wp.func
def friction_delta(
    dp_rel: wp.vec3,   # relative tangential displacement since substep start
    n_hat: wp.vec3,    # contact normal (unit)
    penetration: float,  # positive depth that was resolved this iteration
    mu: float,
) -> wp.vec3:
    """Return the *relative* tangential correction to remove (before splitting
    by inverse mass). Static friction (within the cone) cancels the full
    tangential slide; dynamic friction caps the cancelled amount at ``μ·d``."""
    # Tangential component of the relative motion.
    dp_t = dp_rel - n_hat * wp.dot(dp_rel, n_hat)
    lt = wp.length(dp_t)
    if lt < EPS:
        return wp.vec3(0.0, 0.0, 0.0)
    scale = float(1.0)
    cone = mu * penetration
    if lt >= cone:
        scale = cone / lt  # slipping: only cancel μ·d of the slide
    return dp_t * (-scale)


# -----------------------------------------------------------------------------
# Colored Gauss-Seidel constraint sweep  (Algorithm 1, lines 7-10)
# -----------------------------------------------------------------------------
@wp.kernel
def solve_constraints_color(
    x: wp.array(dtype=wp.vec3),
    x_prev: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    c_type: wp.array(dtype=int),
    c_body_a: wp.array(dtype=int),
    c_body_b: wp.array(dtype=int),
    c_rest: wp.array(dtype=float),
    c_compliance: wp.array(dtype=float),
    c_anchor: wp.array(dtype=wp.vec3),
    c_mu: wp.array(dtype=float),
    c_lambda: wp.array(dtype=float),
    c_active: wp.array(dtype=int),
    # CSR color layout: this launch owns order[color_start : color_start + dim]
    color_order: wp.array(dtype=int),
    color_start: int,
    dt: float,
):
    tid = wp.tid()
    j = color_order[color_start + tid]
    if c_active[j] == 0:
        return

    t = c_type[j]
    inv_dt2 = 1.0 / (dt * dt)

    # ---- DISTANCE & ATTACH: two-sided compliant constraints (Eq. 18, 17) ----
    if t == DISTANCE or t == ATTACH:
        a = c_body_a[j]
        wa = inv_mass[a]
        if t == ATTACH:
            pb = c_anchor[j]
            wb = 0.0
        else:
            b = c_body_b[j]
            pb = x[b]
            wb = inv_mass[b]
        wsum = wa + wb
        if wsum == 0.0:
            return
        d = x[a] - pb
        L = wp.length(d)
        if L < EPS:
            return
        n_hat = d / L
        C = L - c_rest[j]
        alpha = c_compliance[j] * inv_dt2
        dlam = (-C - alpha * c_lambda[j]) / (wsum + alpha)
        c_lambda[j] = c_lambda[j] + dlam
        x[a] = x[a] + n_hat * (wa * dlam)
        if t == DISTANCE:
            x[c_body_b[j]] = x[c_body_b[j]] - n_hat * (wb * dlam)
        return

    # ---- FLOOR: one-sided push-only contact + Coulomb friction --------------
    if t == FLOOR:
        a = c_body_a[j]
        wa = inv_mass[a]
        if wa == 0.0:
            return
        floor_y = c_anchor[j][1]
        C = x[a][1] - floor_y
        if C >= 0.0:
            return  # separated — no contact this iteration
        # Hard normal projection (α = 0): Δλ = −C / w, Δx_y = w·Δλ = −C.
        dlam = -C / wa
        c_lambda[j] = c_lambda[j] + dlam
        x[a] = x[a] + wp.vec3(0.0, wa * dlam, 0.0)
        # Position-based friction against the slide since substep start.
        mu = c_mu[j]
        if mu > 0.0:
            dp_rel = x[a] - x_prev[a]
            corr = friction_delta(dp_rel, wp.vec3(0.0, 1.0, 0.0), -C, mu)
            x[a] = x[a] + corr  # single body: full relative correction applies
        return

    # ---- CONTACT: one-sided sphere-sphere + friction ------------------------
    if t == CONTACT:
        a = c_body_a[j]
        b = c_body_b[j]
        wa = inv_mass[a]
        wb = inv_mass[b]
        wsum = wa + wb
        if wsum == 0.0:
            return
        d = x[a] - x[b]
        L = wp.length(d)
        if L < EPS:
            return
        n_hat = d / L
        C = L - c_rest[j]
        if C >= 0.0:
            return  # not overlapping
        dlam = -C / wsum
        c_lambda[j] = c_lambda[j] + dlam
        x[a] = x[a] + n_hat * (wa * dlam)
        x[b] = x[b] - n_hat * (wb * dlam)
        mu = c_mu[j]
        if mu > 0.0:
            dp_rel = (x[a] - x_prev[a]) - (x[b] - x_prev[b])
            corr = friction_delta(dp_rel, n_hat, -C, mu)
            x[a] = x[a] + corr * (wa / wsum)
            x[b] = x[b] - corr * (wb / wsum)
        return


# -----------------------------------------------------------------------------
# Jacobi constraint sweep  (the XPBD paper's 3D GPU mode, §6)
# -----------------------------------------------------------------------------
# All constraints are processed in one launch; each accumulates its endpoint
# corrections into a shared delta buffer via atomics, then ``apply_jacobi``
# averages by the per-particle constraint count (under-relaxed Jacobi). No
# coloring required, so dynamic contact sets need no per-frame recolor.
@wp.kernel
def solve_constraints_jacobi(
    x: wp.array(dtype=wp.vec3),
    x_prev: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    c_type: wp.array(dtype=int),
    c_body_a: wp.array(dtype=int),
    c_body_b: wp.array(dtype=int),
    c_rest: wp.array(dtype=float),
    c_compliance: wp.array(dtype=float),
    c_anchor: wp.array(dtype=wp.vec3),
    c_mu: wp.array(dtype=float),
    c_lambda: wp.array(dtype=float),
    c_active: wp.array(dtype=int),
    dt: float,
    # outputs (accumulators)
    dx: wp.array(dtype=wp.vec3),
    dn: wp.array(dtype=int),
):
    j = wp.tid()
    if c_active[j] == 0:
        return
    t = c_type[j]
    inv_dt2 = 1.0 / (dt * dt)

    if t == DISTANCE or t == ATTACH:
        a = c_body_a[j]
        wa = inv_mass[a]
        if t == ATTACH:
            pb = c_anchor[j]
            wb = 0.0
        else:
            b = c_body_b[j]
            pb = x[b]
            wb = inv_mass[b]
        wsum = wa + wb
        if wsum == 0.0:
            return
        d = x[a] - pb
        L = wp.length(d)
        if L < EPS:
            return
        n_hat = d / L
        C = L - c_rest[j]
        alpha = c_compliance[j] * inv_dt2
        dlam = (-C - alpha * c_lambda[j]) / (wsum + alpha)
        c_lambda[j] = c_lambda[j] + dlam
        wp.atomic_add(dx, a, n_hat * (wa * dlam))
        wp.atomic_add(dn, a, 1)
        if t == DISTANCE:
            wp.atomic_add(dx, c_body_b[j], -n_hat * (wb * dlam))
            wp.atomic_add(dn, c_body_b[j], 1)
        return

    if t == FLOOR:
        a = c_body_a[j]
        wa = inv_mass[a]
        if wa == 0.0:
            return
        floor_y = c_anchor[j][1]
        C = x[a][1] - floor_y
        if C >= 0.0:
            return
        dlam = -C / wa
        c_lambda[j] = c_lambda[j] + dlam
        corr = wp.vec3(0.0, wa * dlam, 0.0)
        mu = c_mu[j]
        if mu > 0.0:
            dp_rel = (x[a] + corr) - x_prev[a]
            corr = corr + friction_delta(dp_rel, wp.vec3(0.0, 1.0, 0.0), -C, mu)
        wp.atomic_add(dx, a, corr)
        wp.atomic_add(dn, a, 1)
        return

    if t == CONTACT:
        a = c_body_a[j]
        b = c_body_b[j]
        wa = inv_mass[a]
        wb = inv_mass[b]
        wsum = wa + wb
        if wsum == 0.0:
            return
        d = x[a] - x[b]
        L = wp.length(d)
        if L < EPS:
            return
        n_hat = d / L
        C = L - c_rest[j]
        if C >= 0.0:
            return
        dlam = -C / wsum
        c_lambda[j] = c_lambda[j] + dlam
        ca = n_hat * (wa * dlam)
        cb = -n_hat * (wb * dlam)
        mu = c_mu[j]
        if mu > 0.0:
            dp_rel = ((x[a] + ca) - x_prev[a]) - ((x[b] + cb) - x_prev[b])
            fcorr = friction_delta(dp_rel, n_hat, -C, mu)
            ca = ca + fcorr * (wa / wsum)
            cb = cb - fcorr * (wb / wsum)
        wp.atomic_add(dx, a, ca)
        wp.atomic_add(dn, a, 1)
        wp.atomic_add(dx, b, cb)
        wp.atomic_add(dn, b, 1)
        return


@wp.kernel
def apply_jacobi(
    x: wp.array(dtype=wp.vec3),
    dx: wp.array(dtype=wp.vec3),
    dn: wp.array(dtype=int),
    inv_mass: wp.array(dtype=float),
    relax: float,
):
    """Apply averaged Jacobi corrections, then clear the accumulators."""
    i = wp.tid()
    c = dn[i]
    if c > 0 and inv_mass[i] > 0.0:
        x[i] = x[i] + dx[i] * (relax / float(c))
    dx[i] = wp.vec3(0.0, 0.0, 0.0)
    dn[i] = 0


# -----------------------------------------------------------------------------
# Velocity finalize  (Algorithm 1, line 16) + optional viscous damping
# -----------------------------------------------------------------------------
@wp.kernel
def finalize_velocity(
    x: wp.array(dtype=wp.vec3),
    x_prev: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    dt: float,
    damping: float,
    v: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    if inv_mass[i] == 0.0:
        v[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    v[i] = (x[i] - x_prev[i]) / dt * (1.0 - damping)


@wp.kernel
def cap_velocity(v: wp.array(dtype=wp.vec3), max_speed: float):
    """Defensive runaway guard. If a contact snapped a body back from a deep
    penetration, ``v = (x − x_prev)/Δt`` can synthesize a huge speed that the
    next substep projects out by ``v·Δt`` — overshooting through other bodies
    and pumping the next collision with even more energy. Clamping the magnitude
    breaks the loop while keeping the direction."""
    i = wp.tid()
    speed = wp.length(v[i])
    if speed > max_speed:
        v[i] = v[i] * (max_speed / speed)
