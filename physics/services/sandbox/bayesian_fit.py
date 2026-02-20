"""
Bayesian GFD base fit and photometric mass parameter derivation.

Two independent methods for deriving galactic mass parameters:

1. fit_gfd_to_observations_with_bayesian()
   Fits GFD velocity curve to observations via 6-parameter Bayesian
   optimization. Good total mass, unreliable decomposition.

2. derive_mass_parameters_from_photometry()
   Uses photometric scale lengths + luminosity ratios + GFD topology
   to determine all 6 mass parameters analytically. No optimization.
   Requires 3.6um photometric decomposition as input.

IMPORTANT: No unicode characters (Windows charmap constraint).
"""

import math

from scipy.optimize import differential_evolution, minimize
from physics.constants import G, M_SUN, KPC_TO_M, A0, THROAT_YN, HORIZON_YN, FK
from physics.aqual import solve_x as aqual_solve_x
from physics.services.rotation.inference import solve_field_geometry
from physics.services.rotation.photometry import derive_mass_parameters_from_photometry
from physics.services.sandbox.pure_inference import (
    invert_observations,
    fit_mass_model_topology,
)


# =====================================================================
# MASS MODEL -> GFD BASE VELOCITY
# =====================================================================

def _hernquist_enc(r, M, a):
    if M <= 0 or a <= 0 or r <= 0:
        return 0.0
    return M * r * r / ((r + a) * (r + a))


def _disk_enc(r, M, Rd):
    if M <= 0 or Rd <= 0 or r <= 0:
        return 0.0
    x = r / Rd
    if x > 50:
        return M
    return M * (1.0 - (1.0 + x) * math.exp(-x))


def _model_enc(r, Mb, ab, Md, Rd, Mg, Rg):
    return _hernquist_enc(r, Mb, ab) + _disk_enc(r, Md, Rd) + _disk_enc(r, Mg, Rg)


def gfd_velocity(r_kpc, Mb, ab, Md, Rd, Mg, Rg, a0_eff):
    """GFD base velocity at radius r_kpc from the covariant field equation."""
    enc = _model_enc(r_kpc, Mb, ab, Md, Rd, Mg, Rg)
    r_m = r_kpc * KPC_TO_M
    if r_m <= 0 or enc <= 0:
        return 0.0
    gN = G * enc * M_SUN / (r_m * r_m)
    y = gN / a0_eff
    x = aqual_solve_x(y)
    return math.sqrt(a0_eff * x * r_m) / 1000.0


def gfd_velocity_curve(radii, Mb, ab, Md, Rd, Mg, Rg, a0_eff):
    """GFD base velocity at an array of radii."""
    return [gfd_velocity(r, Mb, ab, Md, Rd, Mg, Rg, a0_eff) for r in radii]


# =====================================================================
# FULL COVARIANT ACTION WITH VORTEX TERM
#
# S_ST = int d4x sqrt(-g) [ R/(16*pi*G)
#                          - a0^2/(8*pi*G) * F(|grad(Phi)|^2/a0^2)
#                          - (sigma^2 * a0^2)/(8*pi*G * f(k))
#                            * V_uv * V^uv * exp(-2*Phi/c^2)
#                          + L_matter ]
#
# The vortex V_uv V^uv term in the action is quadratic (energy
# density), so the natural correction is in v^2 = g*r (proportional
# to acceleration * radius), not in v directly.
#
# The antisymmetric field correction is:
#
#   OUTER ARM (r > Rt):
#     f = (r - Rt) / (Renv - Rt)
#     g_vortex = sigma * (4/13) * gN(r) * f^(3/4)
#     v^2 += g_vortex * r    (additive in v^2 space)
#
#   INNER ARM (r < Rt):
#     f_inner = (Rt - r) / Rt
#     Mirror to outer position r_mirror = Rt + f_inner * (Renv - Rt)
#     g_vortex = sigma * (4/13) * gN(r_mirror) * f_inner^(3/4)
#     v^2 -= g_vortex * r    (antisymmetric: opposite sign)
#
# Taking v = sqrt(v^2) naturally produces an ASYMMETRIC velocity
# correction: the sqrt compresses the inner suppression (subtracting
# from large v^2 barely changes v) and amplifies the outer boost
# (adding to smaller v^2 changes v more). This matches observations.
#
# The gN(r) factor provides natural taper: as r increases toward
# Renv, gN decreases because M_enc grows slower than r^2.
# No explicit damping function is needed.
# =====================================================================

def gfd_covariant_velocity(r_kpc, Mb, ab, Md, Rd, Mg, Rg,
                            a0_eff, sigma, R_t, R_env=0):
    """GFD velocity from the full covariant action with vortex term.

    The correction is additive in v^2 space (antisymmetric in
    acceleration). Taking the square root to get velocity naturally
    compresses the inner arm suppression and amplifies the outer
    arm boost, matching the observed asymmetry.

    Parameters
    ----------
    r_kpc : float
        Galactocentric radius in kpc.
    Mb, ab, Md, Rd, Mg, Rg : float
        Mass model parameters (from photometry + topology).
    a0_eff : float
        Effective acceleration scale (m/s^2).
    sigma : float
        Vortex strength (dimensionless coupling).
    R_t : float
        Throat radius in kpc.
    R_env : float
        Envelope radius in kpc (field boundary).

    Returns
    -------
    float
        Circular velocity in km/s.
    """
    v_base = gfd_velocity(r_kpc, Mb, ab, Md, Rd, Mg, Rg, a0_eff)
    if v_base <= 0 or R_t <= 0 or R_env <= R_t:
        return v_base

    STRUCT_FRAC = 4.0 / 13.0
    EXP = 0.75
    L = R_env - R_t

    # Work in SI (m^2/s^2) for unit-correct arithmetic
    v_sq = (v_base * 1000.0) ** 2
    r_m = r_kpc * KPC_TO_M

    if r_kpc >= R_t:
        # Outer arm: positive correction in v^2
        f = (r_kpc - R_t) / L
        if f > 1.0:
            f = 1.0
        enc = _model_enc(r_kpc, Mb, ab, Md, Rd, Mg, Rg)
        if enc > 0:
            gN = G * enc * M_SUN / (r_m * r_m)
            v_sq += sigma * STRUCT_FRAC * gN * (f ** EXP) * r_m

    elif r_kpc > 0:
        # Inner arm: mirror with opposite sign
        f_inner = (R_t - r_kpc) / R_t
        r_mirror = R_t + f_inner * L
        r_mirror_m = r_mirror * KPC_TO_M
        enc_mirror = _model_enc(r_mirror, Mb, ab, Md, Rd, Mg, Rg)
        if enc_mirror > 0:
            gN_mirror = G * enc_mirror * M_SUN / (r_mirror_m * r_mirror_m)
            v_sq -= sigma * STRUCT_FRAC * gN_mirror * (f_inner ** EXP) * r_m

    if v_sq <= 0:
        return 0.0
    return math.sqrt(v_sq) / 1000.0


def gfd_covariant_velocity_curve(radii, Mb, ab, Md, Rd, Mg, Rg,
                                  a0_eff, sigma, R_t, R_env=0):
    """Full covariant GFD velocity at an array of radii."""
    return [gfd_covariant_velocity(r, Mb, ab, Md, Rd, Mg, Rg,
                                    a0_eff, sigma, R_t, R_env)
            for r in radii]


def fit_sigma_from_photometric_model(photometric_mass_model, obs_r, obs_v,
                                      obs_err, a0_eff, R_t, R_env=0):
    """Fit the single free parameter sigma from the covariant action.

    Given the topology-correct mass model (from photometry), find the
    vortex strength sigma that minimizes chi-squared against observations.

    This is a 1-parameter optimization: the mass profile is FIXED (from
    photometry + topology), and only sigma varies. The vortex coupling
    H(r/Rt) is determined by the topology. The result is the measured
    power output from inside the field origin.

    Parameters
    ----------
    photometric_mass_model : dict
        {'bulge': {'M', 'a'}, 'disk': {'M', 'Rd'}, 'gas': {'M', 'Rd'}}
    obs_r : list of float
        Observation radii (kpc).
    obs_v : list of float
        Observed velocities (km/s).
    obs_err : list of float
        Velocity errors (km/s).
    a0_eff : float
        Effective acceleration scale (m/s^2).
    R_t : float
        Throat radius (kpc).
    R_env : float
        Envelope radius (kpc).

    Returns
    -------
    dict with keys:
        'sigma' : float
            Fitted vortex strength.
        'sigma_squared' : float
            Energy density of the vortex field (sigma^2).
        'rms' : float
            RMS velocity residual with vortex correction (km/s).
        'rms_base' : float
            RMS without vortex (sigma=0) for comparison.
        'improvement' : float
            Percentage RMS improvement from the vortex term.
        'v_pred' : list of float
            Predicted velocities at observation radii.
    """
    Mb = photometric_mass_model["bulge"]["M"]
    ab = photometric_mass_model["bulge"]["a"]
    Md = photometric_mass_model["disk"]["M"]
    Rd = photometric_mass_model["disk"]["Rd"]
    Mg = photometric_mass_model["gas"]["M"]
    Rg = photometric_mass_model["gas"]["Rd"]

    n = len(obs_r)

    def rms_cost(sig):
        """Sum of squared residuals (minimizing RMS)."""
        cost = 0.0
        for j in range(n):
            vp = gfd_covariant_velocity(
                obs_r[j], Mb, ab, Md, Rd, Mg, Rg,
                a0_eff, sig, R_t, R_env)
            delta = obs_v[j] - vp
            cost += delta * delta
        return cost

    # sigma is the vortex coupling strength. With the additive v^2
    # correction (proportional to gN * r), sigma needs to be larger
    # than in the multiplicative model because gN*r << v^2_GFD in
    # the deep MOND regime. Typical range: 0 to 15.
    best_sig = 0.0
    best_cost = rms_cost(0.0)
    for s_int in range(-20, 201):
        sig = s_int * 0.1
        cost = rms_cost(sig)
        if cost < best_cost:
            best_cost = cost
            best_sig = sig

    from scipy.optimize import minimize_scalar
    result = minimize_scalar(
        rms_cost,
        bounds=(max(best_sig - 1.0, -5.0), min(best_sig + 1.0, 25.0)),
        method='bounded')
    if result.success:
        best_sig = result.x

    # Compute final velocities and RMS
    v_pred = []
    ss = 0.0
    for j in range(n):
        vp = gfd_covariant_velocity(
            obs_r[j], Mb, ab, Md, Rd, Mg, Rg,
            a0_eff, best_sig, R_t, R_env)
        v_pred.append(round(vp, 2))
        delta = obs_v[j] - vp
        ss += delta * delta
    rms_sigma = math.sqrt(ss / n)

    # Base RMS (no vortex)
    ss_base = 0.0
    for j in range(n):
        vp = gfd_velocity(obs_r[j], Mb, ab, Md, Rd, Mg, Rg, a0_eff)
        delta = obs_v[j] - vp
        ss_base += delta * delta
    rms_base = math.sqrt(ss_base / n)

    improvement = 0.0
    if rms_base > 0:
        improvement = (rms_base - rms_sigma) / rms_base * 100.0

    return {
        "sigma": round(best_sig, 4),
        "sigma_squared": round(best_sig * best_sig, 6),
        "rms": round(rms_sigma, 2),
        "rms_base": round(rms_base, 2),
        "improvement": round(improvement, 1),
        "v_pred": v_pred,
    }


# =====================================================================
# CHI-SQUARED COST FUNCTION
# =====================================================================

def _chi2_cost(params, obs_r, obs_v, obs_w, a0_eff):
    """Weighted chi-squared: sum of w_j * (v_obs - v_pred)^2."""
    Mb, ab, Md, Rd, Mg, Rg = params
    cost = 0.0
    for j in range(len(obs_r)):
        v_pred = gfd_velocity(obs_r[j], Mb, ab, Md, Rd, Mg, Rg, a0_eff)
        delta = obs_v[j] - v_pred
        cost += obs_w[j] * delta * delta
    return cost


def _chi2_cost_topology(params, obs_r, obs_v, obs_w, a0_eff, lam):
    """Chi-squared + maximum entropy regularization on mass fractions.

    The triforce aperture structure (0.30 + 0.30 + 0.30 + 0.10) mandates
    three active mass channels. Velocity chi-squared alone cannot
    distinguish decompositions (the chi-squared surface is nearly flat
    in the mass-fraction direction). The entropy term acts as a
    topological prior: it prevents degenerate solutions where one
    component absorbs all mass, while allowing the velocity data to
    determine the actual fractions.

    Entropy penalty: -sum(log(f_i)) for mass fractions f_i.
    Minimum at equal fractions (1/3, 1/3, 1/3).
    The weight 'lam' (calibrated from Pass 1) controls how strongly
    the topology guides the decomposition.
    """
    Mb, ab, Md, Rd, Mg, Rg = params

    # Velocity chi-squared
    chi2 = 0.0
    for j in range(len(obs_r)):
        v_pred = gfd_velocity(obs_r[j], Mb, ab, Md, Rd, Mg, Rg, a0_eff)
        delta = obs_v[j] - v_pred
        chi2 += obs_w[j] * delta * delta

    # Maximum entropy penalty on mass fractions
    M_total = Mb + Md + Mg
    if M_total > 1e5:
        fb = max(Mb / M_total, 1e-10)
        fd = max(Md / M_total, 1e-10)
        fg = max(Mg / M_total, 1e-10)
        neg_entropy = -(math.log(fb) + math.log(fd) + math.log(fg))
        # Normalized: zero when fractions are equal (1/3 each)
        penalty = neg_entropy - 3.0 * math.log(3.0)
    else:
        penalty = 100.0

    return chi2 + lam * penalty


# =====================================================================
# PARAMETER BOUNDS ESTIMATION
# =====================================================================

def _estimate_bounds(obs_r, obs_v, a0_eff):
    """Estimate reasonable parameter bounds from observation data."""
    r_max = max(obs_r)
    v_max = max(obs_v)

    r_m = r_max * KPC_TO_M
    v_m = v_max * 1000.0
    x = (v_m * v_m) / (a0_eff * r_m)
    y = (x * x) / (1.0 + x)
    M_total_est = y * r_m * r_m * a0_eff / (G * M_SUN)

    M_hi = M_total_est * 2.0

    return [
        (0.0, M_hi),                   # Mb
        (0.01, r_max * 0.3),           # ab
        (0.0, M_hi),                   # Md
        (0.1, r_max * 0.8),            # Rd
        (0.0, M_hi),                   # Mg
        (0.2, r_max * 1.5),            # Rg
    ]


def _topology_bounds(M_total, r_env):
    """Topology-constrained bounds from the triforce aperture structure.

    The triforce face has 3 outer channels + 1 central throat:
      0.30 + 0.30 + 0.30 + 0.10 = 1.0

    R_t = 0.30 * R_env is the throat boundary. All three mass components
    have scale lengths INSIDE R_t (their mass is concentrated within
    the throat, not beyond it):

      Bulge (Hernquist):  a   ~ 0.01 * R_env  (compact, deep inside throat)
      Stellar disk:       Rd  ~ 0.04 * R_env  (intermediate, straddles throat core)
      Gas disk:           Rg  ~ 0.12 * R_env  (extended, fills inner envelope)

    Empirical ranges from the SPARC catalog:
      a  / R_env:  0.003 to 0.03    (median ~ 0.01)
      Rd / R_env:  0.02  to 0.10    (median ~ 0.04)
      Rg / R_env:  0.06  to 0.25    (median ~ 0.12)

    Each mass channel must carry >= 5% of total baryonic mass, enforcing
    the topological requirement that all three channels are active.
    """
    M_min = M_total * 0.05
    M_hi = M_total * 0.90

    return [
        (M_min, M_hi),                              # Mb: 5-90% of total
        (r_env * 0.003, r_env * 0.03),              # ab: 0.3-3% of R_env
        (M_min, M_hi),                              # Md: 5-90% of total
        (r_env * 0.02, r_env * 0.10),               # Rd: 2-10% of R_env
        (M_min, M_hi),                              # Mg: 5-90% of total
        (r_env * 0.06, r_env * 0.25),               # Rg: 6-25% of R_env
    ]


# =====================================================================
# STANDALONE: FIT GFD TO OBSERVATIONS (Pass 1 only)
#
# Fits the GFD covariant field equation directly to observation data
# using a 6-parameter mass model as the optimizer's search space.
# The mass model is an implementation detail: the purpose of this
# method is to find the best GFD velocity curve, not to derive a
# trustworthy mass decomposition.
# =====================================================================

def fit_gfd_to_observations_with_bayesian(obs_r, obs_v, obs_err, a0_eff,
                                          seed=42):
    """Fit the GFD covariant field equation to velocity observations.

    Uses differential_evolution (global) + L-BFGS-B (local polish) to
    find the mass model parameterization that produces the GFD velocity
    curve with the lowest chi-squared against the observed data.

    The optimizer searches a 6-parameter space (Mb, ab, Md, Rd, Mg, Rg)
    because the GFD field equation requires an enclosed mass profile
    M_enc(r) to compute velocities. The 6 parameters are an
    implementation detail: they define the curve, but their individual
    values are NOT a reliable mass decomposition (the chi-squared
    surface is nearly flat in the mass-fraction direction).

    For a reliable mass decomposition, pass the output of this method
    to derive_mass_model_from_bayesian_gfd_fit().

    Parameters
    ----------
    obs_r : list of float
        Observation radii in kpc.
    obs_v : list of float
        Observed circular velocities in km/s.
    obs_err : list of float
        1-sigma velocity errors in km/s.
    a0_eff : float
        Effective acceleration scale in m/s^2 (A0 * accel_ratio).
    seed : int
        Random seed for reproducibility (default 42).

    Returns
    -------
    dict with keys:
        'v_pred' : list of float
            GFD predicted velocity at each observation radius (km/s).
        'params' : tuple of 6 floats
            (Mb, ab, Md, Rd, Mg, Rg) that define the fitted curve.
            Implementation detail, not a reliable decomposition.
        'rms' : float
            RMS velocity residual vs observations (km/s).
        'chi2_dof' : float
            Chi-squared per degree of freedom.
        'n_obs' : int
            Number of observations used.
    """
    n = len(obs_r)
    if n < 3:
        return {"error": "Need at least 3 observations", "n_obs": n}

    obs_w = [1.0 / (e * e) for e in obs_err]

    bounds = _estimate_bounds(obs_r, obs_v, a0_eff)

    result = differential_evolution(
        _chi2_cost,
        bounds=bounds,
        args=(obs_r, obs_v, obs_w, a0_eff),
        seed=seed,
        maxiter=300,
        tol=1e-8,
        popsize=20,
        mutation=(0.5, 1.5),
        recombination=0.8,
        polish=False,
    )

    polished = minimize(
        _chi2_cost,
        result.x,
        args=(obs_r, obs_v, obs_w, a0_eff),
        method='L-BFGS-B',
        bounds=bounds,
    )

    best = polished.x if polished.success else result.x
    chi2_val = polished.fun if polished.success else result.fun

    Mb, ab, Md, Rd, Mg, Rg = best

    # Compute predicted velocities and RMS at observation radii
    v_pred = []
    ss = 0.0
    for j in range(n):
        vp = gfd_velocity(obs_r[j], Mb, ab, Md, Rd, Mg, Rg, a0_eff)
        v_pred.append(round(vp, 2))
        delta = obs_v[j] - vp
        ss += delta * delta
    rms = math.sqrt(ss / n) if n > 0 else 0.0
    chi2_dof = chi2_val / max(n - 6, 1)

    return {
        "v_pred": v_pred,
        "params": tuple(float(p) for p in best),
        "rms": round(rms, 2),
        "chi2_dof": round(chi2_dof, 4),
        "n_obs": n,
    }


# derive_mass_parameters_from_photometry: see physics.services.rotation.photometry


# =====================================================================
# FULL BAYESIAN FIT (two-pass: geometry then topology decomposition)
#
# Orchestrator that calls fit_gfd_to_observations_with_bayesian()
# for Pass 1, then applies topology constraints in Pass 2, generates
# chart data, and compares mass models.
# =====================================================================

def fit_gfd_bayesian(obs_r, obs_v, obs_err, a0_eff, num_points=500,
                     chart_max_r=None, seed=42, photo_scales=None):
    """Fit GFD velocity curve to observations.

    Single-pass differential_evolution + L-BFGS-B polish.
    When photometric scale lengths are available, they anchor the
    bounds so the optimizer can't collapse mass into unphysical
    configurations. Components are allowed to be zero (no forced
    minimum mass fractions).

    Returns dict with fitted parameters, GFD base curve, RMS,
    per-point residuals, and field geometry.
    """
    n = len(obs_r)

    # Cap errors at 6.2% of observed velocity so no observation
    # is treated as completely unconstrained by the optimizer.
    MAX_ERR_FRAC = 0.062
    obs_err_capped = [min(e, v * MAX_ERR_FRAC) for e, v in
                       zip(obs_err, obs_v)]
    obs_w = [1.0 / (e * e) for e in obs_err_capped]

    # Build bounds: data-driven, refined by photometric priors
    bounds = _estimate_bounds(obs_r, obs_v, a0_eff)

    if photo_scales:
        ps_ab = photo_scales.get("ab", 0)
        ps_Rd = photo_scales.get("Rd", 0)
        ps_Rg = photo_scales.get("Rg", 0)
        if ps_ab > 0:
            bounds[1] = (max(0.01, ps_ab * 0.5), ps_ab * 2.0)
        if ps_Rd > 0:
            bounds[3] = (max(0.05, ps_Rd * 0.5), ps_Rd * 2.0)
        if ps_Rg > 0:
            bounds[5] = (max(0.1, ps_Rg * 0.5), ps_Rg * 2.0)

    result = differential_evolution(
        _chi2_cost,
        bounds=bounds,
        args=(obs_r, obs_v, obs_w, a0_eff),
        seed=seed,
        maxiter=500,
        tol=1e-9,
        popsize=30,
        mutation=(0.5, 1.5),
        recombination=0.8,
        polish=False,
    )

    polished = minimize(
        _chi2_cost,
        result.x,
        args=(obs_r, obs_v, obs_w, a0_eff),
        method='L-BFGS-B',
        bounds=bounds,
    )

    best = polished.x if polished.success else result.x
    chi2_final = polished.fun if polished.success else result.fun

    best = [max(lo, min(hi, v))
            for v, (lo, hi) in zip(best, bounds)]

    Mb, ab, Md, Rd, Mg, Rg = best

    # Per-point residuals at observation radii
    residuals = []
    ss = 0.0
    for j in range(n):
        v_pred = gfd_velocity(obs_r[j], Mb, ab, Md, Rd, Mg, Rg, a0_eff)
        delta = obs_v[j] - v_pred
        ss += delta * delta
        residuals.append({
            "r": obs_r[j],
            "v_obs": obs_v[j],
            "v_gfd": round(v_pred, 2),
            "delta": round(delta, 2),
            "err": obs_err[j],
        })
    rms = math.sqrt(ss / n) if n > 0 else 0.0
    chi2_dof = chi2_final / max(n - 6, 1)

    # Derive field geometry from the topology-constrained model
    geom = solve_field_geometry(Mb, ab, Md, Rd, Mg, Rg, a0_eff)

    # Generate smooth GFD base curve for charting.
    # The x-axis displays to R_env * 1.10, so the curve data must
    # extend past that to avoid the line stopping short.
    derived_renv = geom.get("envelope_radius_kpc") or 0
    r_max = max(
        chart_max_r if chart_max_r else 0,
        derived_renv * 1.15,
        max(obs_r) * 1.15)
    dr = r_max / num_points
    chart_radii = [dr * (i + 1) for i in range(num_points)]
    chart_velocities = gfd_velocity_curve(
        chart_radii, Mb, ab, Md, Rd, Mg, Rg, a0_eff)

    # =================================================================
    # MASS MODEL: directly from the topology-constrained optimizer
    #
    # The two-pass Bayesian optimizer (Pass 2) produces the mass
    # decomposition directly. No post-hoc grid search needed: the
    # optimizer found the best (Mb, ab, Md, Rd, Mg, Rg) within
    # topology-constrained bounds derived from R_env.
    # =================================================================
    bayesian_mass_model = {
        "bulge": {"M": round(Mb, 2), "a": round(ab, 4)},
        "disk": {"M": round(Md, 2), "Rd": round(Rd, 4)},
        "gas": {"M": round(Mg, 2), "Rd": round(Rg, 4)},
    }
    bayesian_M_total = round(Mb + Md + Mg, 2)

    # =================================================================
    # OBSERVATION-INVERTED MASS MODEL (for comparison)
    #
    # Invert raw observations through the field equation to get
    # M_enc(r), then fit a 3-component model using the topology-
    # aware grid search. This uses the noisy observation data
    # directly, without the Bayesian curve smoothing.
    # Comparing the two reveals how much the mass decomposition
    # is affected by observational scatter vs. the smooth fit.
    # =================================================================
    obs_dicts = [
        {"r": obs_r[j], "v": obs_v[j], "err": obs_err[j]}
        for j in range(n)
    ]
    inverted_profile_obs = invert_observations(obs_dicts, a0_eff)
    obs_mass_fit = fit_mass_model_topology(
        inverted_profile_obs, a0_eff, derived_renv)

    obs_mass_model = None
    obs_M_total = 0.0
    if obs_mass_fit:
        obs_mass_model = obs_mass_fit["model"]
        ob = obs_mass_model.get("bulge", {})
        od = obs_mass_model.get("disk", {})
        og = obs_mass_model.get("gas", {})
        obs_M_total = (
            ob.get("M", 0) + od.get("M", 0) + og.get("M", 0)
        )

    # Compute % delta between the two mass models
    def _pct_delta(obs_val, bay_val):
        """Percentage difference: (obs - bay) / bay * 100."""
        if bay_val is None or obs_val is None:
            return None
        if abs(bay_val) < 1e-10:
            if abs(obs_val) < 1e-10:
                return 0.0
            return None
        return round((obs_val - bay_val) / abs(bay_val) * 100.0, 1)

    mass_comparison = None
    if obs_mass_model:
        ob = obs_mass_model.get("bulge", {})
        od = obs_mass_model.get("disk", {})
        og = obs_mass_model.get("gas", {})
        bb = bayesian_mass_model.get("bulge", {})
        bd = bayesian_mass_model.get("disk", {})
        bg = bayesian_mass_model.get("gas", {})
        mass_comparison = {
            "gas": {
                "obs_M": og.get("M", 0),
                "bay_M": bg.get("M", 0),
                "delta_M": _pct_delta(og.get("M", 0), bg.get("M", 0)),
                "obs_Rd": og.get("Rd", 0),
                "bay_Rd": bg.get("Rd", 0),
                "delta_Rd": _pct_delta(og.get("Rd", 0), bg.get("Rd", 0)),
            },
            "disk": {
                "obs_M": od.get("M", 0),
                "bay_M": bd.get("M", 0),
                "delta_M": _pct_delta(od.get("M", 0), bd.get("M", 0)),
                "obs_Rd": od.get("Rd", 0),
                "bay_Rd": bd.get("Rd", 0),
                "delta_Rd": _pct_delta(od.get("Rd", 0), bd.get("Rd", 0)),
            },
            "bulge": {
                "obs_M": ob.get("M", 0),
                "bay_M": bb.get("M", 0),
                "delta_M": _pct_delta(ob.get("M", 0), bb.get("M", 0)),
                "obs_a": ob.get("a", 0),
                "bay_a": bb.get("a", 0),
                "delta_a": _pct_delta(ob.get("a", 0), bb.get("a", 0)),
            },
            "total": {
                "obs_M": round(obs_M_total, 2),
                "bay_M": bayesian_M_total,
                "delta_M": _pct_delta(obs_M_total, bayesian_M_total),
            },
        }

    return {
        "mass_model": bayesian_mass_model,
        "obs_mass_model": obs_mass_model,
        "mass_comparison": mass_comparison,
        "M_total": bayesian_M_total,
        "obs_M_total": round(obs_M_total, 2) if obs_mass_model else None,
        "rms": round(rms, 2),
        "chi2_dof": round(chi2_dof, 4),
        "n_obs": n,
        "residuals": residuals,
        "field_geometry": geom,
        "chart": {
            "radii": [round(r, 4) for r in chart_radii],
            "gfd_base": [round(v, 4) for v in chart_velocities],
        },
    }


# =====================================================================
# FAST GFD FIT (L-BFGS-B from photometric starting point)
#
# Uses the photometric mass parameters as the initial guess and runs
# a quick local optimizer (L-BFGS-B) to find the best-fit GFD curve.
# Same physics as the full Bayesian but ~100x faster because it skips
# the global differential_evolution search.
#
# The result is a smooth GFD curve through the observations, plus the
# vortex delta against the photometric curve and diagnostics.
# =====================================================================

def fit_observation_with_spline_then_gfd_bayesian(
        obs_r, obs_v, obs_err, chart_radii, photo_vels, a0_eff,
        photo_params=None):
    """Fast GFD fit: L-BFGS-B from photometric starting point.

    Uses the same GFD field equation and 6.2% error cap as the full
    Bayesian, but starts from the photometric parameters and uses a
    local optimizer only. Completes in ~100-300ms instead of 9-12s.

    Parameters
    ----------
    obs_r : list of float
        Observation radii (kpc).
    obs_v : list of float
        Observed circular velocities (km/s).
    obs_err : list of float
        Velocity errors (km/s).
    chart_radii : list of float
        Dense uniform radii for the output curves.
    photo_vels : list of float
        Photometric GFD velocities evaluated at chart_radii.
    a0_eff : float
        Effective acceleration scale (m/s^2).
    photo_params : dict or None
        Photometric mass parameters:
        {'Mb': float, 'ab': float, 'Md': float, 'Rd': float,
         'Mg': float, 'Rg': float}

    Returns
    -------
    dict with keys:
        'spline_vels' : list of float
            Fast-fit GFD velocity at each chart radius (km/s).
        'gfd_covariant_spline' : list of float
            Same as spline_vels (for API compatibility).
        'delta_v2' : list of float
            v_fit^2 - v_photo^2 at each chart radius.
        'vortex_signal' : dict
            sigma_net, energy_boost, energy_suppress, energy_ratio.
        'rms' : float
            RMS of (v_obs - v_fit) at observation radii (km/s).
        'n_obs' : int
            Number of observations used.
    """
    n = len(obs_r)
    if n < 3 or not photo_params or not a0_eff:
        return {"error": "Need >= 3 observations and photo_params", "n_obs": n}

    MAX_ERR_FRAC = 0.062
    obs_err_capped = [min(e, v * MAX_ERR_FRAC) for e, v in
                       zip(obs_err, obs_v)]
    obs_w = [1.0 / (e * e) for e in obs_err_capped]

    Mb0 = photo_params["Mb"]
    ab0 = photo_params["ab"]
    Md0 = photo_params["Md"]
    Rd0 = photo_params["Rd"]
    Mg0 = photo_params["Mg"]
    Rg0 = photo_params["Rg"]

    # Same bounds strategy as the full Bayesian: data-driven mass
    # bounds + tight photometric scale length constraints
    bounds = _estimate_bounds(obs_r, obs_v, a0_eff)
    if ab0 > 0:
        bounds[1] = (max(0.01, ab0 * 0.5), ab0 * 2.0)
    if Rd0 > 0:
        bounds[3] = (max(0.05, Rd0 * 0.5), Rd0 * 2.0)
    if Rg0 > 0:
        bounds[5] = (max(0.1, Rg0 * 0.5), Rg0 * 2.0)

    # Mini differential_evolution: same algorithm as the full
    # Bayesian but with smaller population and fewer iterations.
    # Tight photometric bounds make convergence fast.
    de_result = differential_evolution(
        _chi2_cost,
        bounds=bounds,
        args=(obs_r, obs_v, obs_w, a0_eff),
        seed=42,
        maxiter=100,
        tol=1e-8,
        popsize=10,
        mutation=(0.5, 1.5),
        recombination=0.8,
        polish=False,
    )

    polished = minimize(
        _chi2_cost,
        de_result.x,
        args=(obs_r, obs_v, obs_w, a0_eff),
        method='L-BFGS-B',
        bounds=bounds,
    )

    best = polished.x if polished.success else de_result.x
    Mb, ab, Md, Rd, Mg, Rg = best

    # Generate smooth fit curve on chart radii
    fit_vels = gfd_velocity_curve(chart_radii, Mb, ab, Md, Rd, Mg, Rg, a0_eff)

    # Compute delta_v2 and covariant at chart radii
    spline_at_chart = []
    delta_v2 = []
    covariant_spline = []

    for i in range(len(chart_radii)):
        vf = fit_vels[i]
        vp = photo_vels[i] if i < len(photo_vels) else 0.0
        d = vf * vf - vp * vp

        spline_at_chart.append(round(vf, 4))
        delta_v2.append(round(d, 2))
        covariant_spline.append(round(vf, 4))

    # RMS and vortex diagnostics at observation points
    sum_delta_pos = 0.0
    sum_delta_neg = 0.0
    sum_v2_photo = 0.0
    ss = 0.0

    for j in range(n):
        vf = gfd_velocity(obs_r[j], Mb, ab, Md, Rd, Mg, Rg, a0_eff)
        vp = _interp_linear(obs_r[j], chart_radii, photo_vels)
        d = vf * vf - vp * vp

        if d > 0:
            sum_delta_pos += d
        else:
            sum_delta_neg += d
        sum_v2_photo += vp * vp

        delta_obs = obs_v[j] - vf
        ss += delta_obs * delta_obs

    rms = math.sqrt(ss / n) if n > 0 else 0.0

    sigma_net = (
        (sum_delta_pos + sum_delta_neg) / sum_v2_photo
        if sum_v2_photo > 0 else 0.0)

    return {
        "spline_vels": spline_at_chart,
        "gfd_covariant_spline": covariant_spline,
        "delta_v2": delta_v2,
        "vortex_signal": {
            "sigma_net": round(sigma_net, 4),
            "energy_boost": round(sum_delta_pos, 0),
            "energy_suppress": round(sum_delta_neg, 0),
            "energy_ratio": round(
                abs(sum_delta_pos / sum_delta_neg)
                if sum_delta_neg != 0 else 0, 2),
        },
        "rms": round(rms, 2),
        "n_obs": n,
    }


# =====================================================================
# FAST GFD FIT WITH ACCELERATION (7-parameter)
#
# Same as fit_observation_with_spline_then_gfd_bayesian but adds
# accel_ratio as a 7th free parameter. This allows the effective a0
# to vary per galaxy, diagnosing whether the mass model input is
# under/over-estimating the enclosed mass.
# =====================================================================

def _chi2_cost_with_accel(params, obs_r, obs_v, obs_w, a0_base):
    """Chi2 cost with accel_ratio as 7th parameter."""
    Mb, ab, Md, Rd, Mg, Rg, accel_ratio = params
    a0_eff = a0_base * accel_ratio
    cost = 0.0
    for j in range(len(obs_r)):
        v_pred = gfd_velocity(obs_r[j], Mb, ab, Md, Rd, Mg, Rg, a0_eff)
        delta = obs_v[j] - v_pred
        cost += obs_w[j] * delta * delta
    return cost


def fit_observation_with_spline_then_gfd_bayesian_with_acceleration(
        obs_r, obs_v, obs_err, chart_radii, photo_vels, a0_eff,
        photo_params=None):
    """Fast GFD fit with acceleration as a 7th free parameter.

    Identical to fit_observation_with_spline_then_gfd_bayesian but
    adds accel_ratio to the optimizer. The fitted accel_ratio reveals
    how much the effective a0 must shift to match observations,
    diagnosing mass model input errors.

    Parameters
    ----------
    obs_r, obs_v, obs_err : lists of float
        Observation data.
    chart_radii : list of float
        Dense radii for output curves.
    photo_vels : list of float
        Photometric GFD velocities at chart_radii.
    a0_eff : float
        Base effective acceleration scale (m/s^2).
    photo_params : dict or None
        Photometric mass parameters.

    Returns
    -------
    dict with keys:
        'accel_vels' : list of float
            Fitted GFD velocity at each chart radius (km/s).
        'delta_v2' : list of float
            v_fit^2 - v_photo^2 at each chart radius.
        'vortex_signal' : dict
            sigma_net, energy_boost, energy_suppress, energy_ratio.
        'rms' : float
            RMS of (v_obs - v_fit) at observation radii (km/s).
        'accel_ratio' : float
            Fitted acceleration ratio (1.0 = universal a0).
        'n_obs' : int
            Number of observations used.
    """
    n = len(obs_r)
    if n < 3 or not photo_params or not a0_eff:
        return {"error": "Need >= 3 observations and photo_params", "n_obs": n}

    MAX_ERR_FRAC = 0.062
    obs_err_capped = [min(e, v * MAX_ERR_FRAC) for e, v in
                       zip(obs_err, obs_v)]
    obs_w = [1.0 / (e * e) for e in obs_err_capped]

    Mb0 = photo_params["Mb"]
    ab0 = photo_params["ab"]
    Md0 = photo_params["Md"]
    Rd0 = photo_params["Rd"]
    Mg0 = photo_params["Mg"]
    Rg0 = photo_params["Rg"]

    # 6 mass bounds (same as spline method) + accel_ratio bounds
    bounds = _estimate_bounds(obs_r, obs_v, a0_eff)
    if ab0 > 0:
        bounds[1] = (max(0.01, ab0 * 0.5), ab0 * 2.0)
    if Rd0 > 0:
        bounds[3] = (max(0.05, Rd0 * 0.5), Rd0 * 2.0)
    if Rg0 > 0:
        bounds[5] = (max(0.1, Rg0 * 0.5), Rg0 * 2.0)
    # accel_ratio: allow 0.3x to 3.0x of universal a0
    bounds.append((0.3, 3.0))

    de_result = differential_evolution(
        _chi2_cost_with_accel,
        bounds=bounds,
        args=(obs_r, obs_v, obs_w, a0_eff),
        seed=42,
        maxiter=100,
        tol=1e-8,
        popsize=10,
        mutation=(0.5, 1.5),
        recombination=0.8,
        polish=False,
    )

    polished = minimize(
        _chi2_cost_with_accel,
        de_result.x,
        args=(obs_r, obs_v, obs_w, a0_eff),
        method='L-BFGS-B',
        bounds=bounds,
    )

    best = polished.x if polished.success else de_result.x
    Mb, ab, Md, Rd, Mg, Rg, accel_ratio = best
    a0_fitted = a0_eff * accel_ratio

    # Generate smooth fit curve on chart radii
    fit_vels = gfd_velocity_curve(
        chart_radii, Mb, ab, Md, Rd, Mg, Rg, a0_fitted)

    accel_at_chart = []
    delta_v2 = []

    for i in range(len(chart_radii)):
        vf = fit_vels[i]
        vp = photo_vels[i] if i < len(photo_vels) else 0.0
        d = vf * vf - vp * vp
        accel_at_chart.append(round(vf, 4))
        delta_v2.append(round(d, 2))

    # RMS and vortex diagnostics at observation points
    sum_delta_pos = 0.0
    sum_delta_neg = 0.0
    sum_v2_photo = 0.0
    ss = 0.0

    for j in range(n):
        vf = gfd_velocity(
            obs_r[j], Mb, ab, Md, Rd, Mg, Rg, a0_fitted)
        vp = _interp_linear(obs_r[j], chart_radii, photo_vels)
        d = vf * vf - vp * vp

        if d > 0:
            sum_delta_pos += d
        else:
            sum_delta_neg += d
        sum_v2_photo += vp * vp

        delta_obs = obs_v[j] - vf
        ss += delta_obs * delta_obs

    rms = math.sqrt(ss / n) if n > 0 else 0.0

    sigma_net = (
        (sum_delta_pos + sum_delta_neg) / sum_v2_photo
        if sum_v2_photo > 0 else 0.0)

    return {
        "accel_vels": accel_at_chart,
        "delta_v2": delta_v2,
        "vortex_signal": {
            "sigma_net": round(sigma_net, 4),
            "energy_boost": round(sum_delta_pos, 0),
            "energy_suppress": round(sum_delta_neg, 0),
            "energy_ratio": round(
                abs(sum_delta_pos / sum_delta_neg)
                if sum_delta_neg != 0 else 0, 2),
        },
        "rms": round(rms, 2),
        "accel_ratio": round(float(accel_ratio), 4),
        "n_obs": n,
    }


def _interp_linear(x, xs, ys):
    """Simple linear interpolation for a single x in sorted xs."""
    if not xs or not ys:
        return 0.0
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo = 0
    hi = len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    t = (x - xs[lo]) / (xs[hi] - xs[lo]) if xs[hi] != xs[lo] else 0.0
    return ys[lo] + t * (ys[hi] - ys[lo])
