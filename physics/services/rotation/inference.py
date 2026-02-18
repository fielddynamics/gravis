"""
Rotation Curve Inference Pipeline for GRAVIS.

Determines the Origin Throughput and refines the baryonic mass model for
a galaxy by fitting the GFD-sigma curve to observed rotation velocities.

Architecture
------------
This module is the canonical location for rotation curve inference logic.
It is dependency-injected into RotationService and serves the
/api/rotation/inference endpoint. No other module should contain inference
optimization code (see physics/engine.py module docstring for the full
architectural pattern).

Pipeline Overview
-----------------
The pipeline runs in three sequential stages:

  Stage 1 (Mass Baseline):
      Accept the published photometric mass model as the starting point.
      The mass model consists of three independently measured components
      (stellar bulge, stellar disk, gas disk), each with a mass and
      spatial scale parameter. These values come from photometry (NIR,
      3.6um), dynamical models, and 21-cm radio surveys. They are NOT
      fitted to rotation curves, preserving their independence as priors.

  Stage 2 (Origin Throughput):
      With the mass model fixed, perform a two-pass grid search over the
      Origin Throughput parameter to find the value that minimizes the
      chi-squared residual between the GFD-sigma curve and observed
      velocities. The search is:
        Pass 1: Coarse sweep from -3.0 to +3.0 in steps of 0.1 (61 evaluations)
        Pass 2: Fine sweep +/- 0.10 around the coarse optimum in steps of 0.01

  Stage 3 (Mass Decomposition):
      Iteratively refine the stellar mass using the sigma curve's
      structural acceleration to back-calculate the enclosed baryonic
      mass at each observation point. Gas mass is held fixed (from 21-cm
      measurements); the stellar mass (bulge + disk) is updated and split
      using the published photometric bulge-to-total ratio. Throughput is
      re-optimized at each iteration. Convergence is checked by monitoring
      the relative change in total stellar mass (tolerance: 1%).

Performance
-----------
Typical execution time: 50 to 150 ms (500 radial sample points, 20 observations).
No external optimizer (scipy) is required. All optimization is grid-based,
exploiting the smoothness of the chi-squared landscape over a 1D parameter.

IMPORTANT: No unicode characters allowed (Windows charmap constraint).
"""

import math

from physics.equations import aqual_solve_x, G, M_SUN, A0, KPC_TO_M
from physics.constants import K_SIMPLEX, THROAT_YN as _THROAT_YN_CONST
from physics.constants import HORIZON_YN as _HORIZON_YN_CONST
from physics.constants import THROAT_FRAC as _THROAT_FRAC_CONST
from physics.sigma import GfdSymmetricStage, auto_vortex_strength

# =====================================================================
# TOPOLOGICAL MANIFOLD CONDITIONS
#
# Two parameter-free conditions define the complete manifold geometry
# for any galaxy, derived from the stellated octahedron topology:
#
# Throat: G * M(<Rt) / Rt^2 = a0 * 18/65
#   yN(Rt) = (4/13)(9/10) = 18/65
#   Structural fraction (4/13) times throughput factor (9/10).
#
# Horizon: G * M(<Renv) / Renv^2 = a0 * 36/1365
#   yN(Renv) = (18/65)(2/21) = 36/1365
#   Throat condition divided by f(k)/2 where f(k) = 1+k+k^2 = 21.
#
# The acceleration drops by f(k)/2 = 10.5 between throat and horizon.
# The radius ratio Renv/Rt ~ sqrt(10.5) ~ 3.24, which becomes 3.33
# (i.e. 1/0.30) when accounting for gas mass beyond the throat.
#
# The 0.30 throat fraction is not assumed. It emerges from the ratio
# of the two yN thresholds applied to the baryonic mass profile.
#
# Both conditions trace every constant to k = 4 (simplex number).
# No dark matter, no free parameters, no fitting.
#
# See: papers/system_architecture/field_geometry/
#      HORIZON_ACCELERATION_CONDITION.md
# =====================================================================
THROAT_YN = _THROAT_YN_CONST    # 18/65 = 0.27692...
HORIZON_YN = _HORIZON_YN_CONST  # 36/1365 = 0.026374...
THROAT_FRAC = _THROAT_FRAC_CONST  # 0.30 (verification, not input)


def solve_field_geometry(Mb, ab, Md, Rd, Mg, Rg, a0_eff):
    """
    Derive the Field Origin (R_t) and Field Horizon (R_env) from the
    baryonic mass model using topological conditions.

    Every galaxy is a vortex. Every vortex has a throat and a horizon.

    For 3-cycle galaxies (yN reaches 18/65):
      Both R_t and R_env are solved independently from their yN
      conditions. R_t/R_env ~ 0.30 emerges and is verified.

    For 2-cycle galaxies (yN never reaches 18/65):
      R_env is solved from yN = 36/1365 on the descending side of
      the yN profile. R_t = 0.30 * R_env from the topological
      constant. The throat exists but at reduced yN (partial closure).

    Parameters
    ----------
    Mb, ab : float
        Bulge mass (Msun) and Hernquist scale (kpc).
    Md, Rd : float
        Disk mass (Msun) and exponential scale length (kpc).
    Mg, Rg : float
        Gas mass (Msun) and exponential scale length (kpc).
    a0_eff : float
        Effective acceleration scale in m/s^2 (A0 * accel_ratio).

    Returns
    -------
    dict with keys:
        'throat_radius_kpc' : float or None
            R_t from yN = 18/65 (3-cycle) or 0.30 * R_env (2-cycle).
            None only if the galaxy has no resolvable horizon.
        'envelope_radius_kpc' : float or None
            R_env from yN = 36/1365. None only if yN never
            reaches the horizon threshold (negligible mass).
        'yN_at_throat' : float
            yN at the solved R_t. Equal to 18/65 for 3-cycle,
            less than 18/65 for 2-cycle.
        'yN_at_horizon' : float
            Verification: yN at solved R_env (~36/1365).
        'throat_fraction' : float or None
            R_t / R_env (~0.30 for 3-cycle, exactly 0.30 for 2-cycle).
        'cycle' : int
            3 if yN reaches 18/65 (full closure), 2 otherwise.
    """
    def _enc(r):
        enc = 0.0
        if Mb > 0 and ab > 0:
            enc += Mb * r * r / ((r + ab) * (r + ab))
        if Md > 0 and Rd > 0:
            x = r / Rd
            enc += Md * (1.0 - (1.0 + x) * math.exp(-x))
        if Mg > 0 and Rg > 0:
            x = r / Rg
            enc += Mg * (1.0 - (1.0 + x) * math.exp(-x))
        return enc

    def _yN(r):
        enc = _enc(r)
        r_m = r * KPC_TO_M
        if r_m <= 0 or enc <= 0:
            return 0.0
        return G * enc * M_SUN / (r_m * r_m * a0_eff)

    def _solve_yN_crossing(target, lo=0.01, hi=500.0):
        """Binary search for radius where yN(r) = target."""
        for _ in range(120):
            mid = (lo + hi) / 2.0
            if _yN(mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    # Find the peak yN to determine if the throat condition is met
    # and to anchor the horizon search on the descending side.
    yN_peak = 0.0
    r_peak = 0.01
    for i in range(1, 2001):
        r = 500.0 / 2000.0 * i
        y = _yN(r)
        if y > yN_peak:
            yN_peak = y
            r_peak = r

    is_3cycle = yN_peak >= THROAT_YN

    # -----------------------------------------------------------------
    # HORIZON: yN(R_env) = 36/1365
    #
    # Solved for ALL galaxies. The horizon exists whenever yN_peak
    # exceeds the horizon threshold. For galaxies too small even
    # for a horizon (negligible mass), return None.
    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    # VISIBLE HORIZON: R_vis = radius enclosing 95% of baryonic mass.
    # This is where the matter physically resides, independent of the
    # gravitational field threshold. For gas-dominated dwarfs, R_vis
    # can be much larger than R_env.
    # -----------------------------------------------------------------
    M_total = _enc(2000.0)
    r_vis = None
    r_vis_90 = None
    r_vis_99 = None
    if M_total > 0:
        for frac, label in [(0.90, "90"), (0.95, "95"), (0.995, "99")]:
            target_enc = frac * M_total
            lo_v, hi_v = 0.01, 2000.0
            for _ in range(120):
                mid_v = (lo_v + hi_v) / 2.0
                if _enc(mid_v) < target_enc:
                    lo_v = mid_v
                else:
                    hi_v = mid_v
            val = round((lo_v + hi_v) / 2.0, 4)
            if label == "90":
                r_vis_90 = val
            elif label == "95":
                r_vis = val
            else:
                r_vis_99 = val

    if yN_peak < HORIZON_YN:
        return {
            "throat_radius_kpc": None,
            "envelope_radius_kpc": None,
            "visible_radius_kpc": r_vis,
            "visible_radius_90_kpc": r_vis_90,
            "visible_radius_99_kpc": r_vis_99,
            "yN_at_throat": yN_peak,
            "yN_at_horizon": 0.0,
            "throat_fraction": None,
            "cycle": 2,
        }

    # Solve horizon on the descending side of the yN profile.
    r_env = _solve_yN_crossing(HORIZON_YN, lo=r_peak, hi=2000.0)

    if is_3cycle:
        # =============================================================
        # 3-CYCLE: full closure. Solve R_t independently from yN = 18/65.
        # R_t/R_env ~ 0.30 emerges and is verified, not assumed.
        # =============================================================
        r_t = _solve_yN_crossing(THROAT_YN)
        throat_frac = round(r_t / r_env, 4) if r_env > 0 else None

        return {
            "throat_radius_kpc": round(r_t, 4),
            "envelope_radius_kpc": round(r_env, 4),
            "visible_radius_kpc": r_vis,
            "visible_radius_90_kpc": r_vis_90,
            "visible_radius_99_kpc": r_vis_99,
            "yN_at_throat": round(_yN(r_t), 6),
            "yN_at_horizon": round(_yN(r_env), 6),
            "throat_fraction": throat_frac,
            "cycle": 3,
        }
    else:
        # =============================================================
        # 2-CYCLE: partial closure. yN never reaches 18/65.
        # The throat still exists (every vortex has a throat).
        # R_t = 0.30 * R_env from the topological constant.
        # =============================================================
        r_t = THROAT_FRAC * r_env

        return {
            "throat_radius_kpc": round(r_t, 4),
            "envelope_radius_kpc": round(r_env, 4),
            "visible_radius_kpc": r_vis,
            "visible_radius_90_kpc": r_vis_90,
            "visible_radius_99_kpc": r_vis_99,
            "yN_at_throat": round(_yN(r_t), 6),
            "yN_at_horizon": round(_yN(r_env), 6),
            "throat_fraction": THROAT_FRAC,
            "cycle": 2,
        }


def optimize_inference(mass_model, max_radius, num_points,
                       observations, accel_ratio,
                       galactic_radius_kpc):
    """
    Three-stage inference pipeline for galactic rotation curves.

    Given a galaxy's published baryonic mass model and a set of observed
    rotation velocities, this function determines the Origin Throughput
    (the single parameter governing the structural correction in
    GFD-sigma) and iteratively refines the mass decomposition.

    R_t and R_env are derived internally from the mass model using the
    topological yN conditions (yN = 18/65 for the throat, yN = 36/1365
    for the horizon). No external galactic_radius is used in the
    inference path. The derived geometry is re-computed after Stage 3
    mass refinement so the sigma stage always uses self-consistent
    values. See THROAT_AND_HORIZON_RADIUS_DERIVATION.md.

    Parameters
    ----------
    mass_model : dict
        Published photometric mass model with keys 'bulge', 'disk', 'gas'.
        Each sub-dict contains 'M' (solar masses) and a scale parameter
        ('a' for bulge Hernquist profile, 'Rd' for disk/gas exponential).
        These values originate from independent measurements (photometry,
        radio surveys), not from rotation curve fitting.
    max_radius : float
        Maximum chart radius in kpc. Defines the outer edge of the radial
        sampling grid.
    num_points : int
        Number of evenly spaced radial sample points. Higher values give
        finer interpolation at the cost of computation time. Typical: 500.
    observations : list of dict
        Observed rotation curve data points. Each dict must contain:
          'r' : float, galactocentric radius in kpc
          'v' : float, circular velocity in km/s
          'err' : float (optional), 1-sigma measurement uncertainty in km/s.
                  Defaults to 5.0 km/s if not provided. Minimum floor: 1.0 km/s.
        At least 3 valid observations (r > 0, v > 0) are required.
    accel_ratio : float
        Acceleration scale ratio relative to a0 = 1.2e-10 m/s^2.
        Default for most galaxies: 1.0.
    galactic_radius_kpc : float
        Retained for backward compatibility. Not used in the inference
        path; R_env is derived from the mass model via solve_field_geometry().

    Returns
    -------
    dict with keys:
        'mass_model' : dict
            Refined mass model (same structure as input, with updated
            masses from Stage 3 decomposition).
        'throughput' : float
            Best-fit Origin Throughput value.
        'gfd_rms' : float
            RMS residual (km/s) of the GFD base curve vs observations.
        'rms' : float
            RMS residual (km/s) of the GFD-sigma curve vs observations.
        'chi2_dof' : float
            Weighted chi-squared per degree of freedom for the sigma curve.
        'method' : str
            Always 'inference' for this pipeline.
        'gene_report' : list of dict
            Per-parameter comparison of published vs fitted values with
            sigma excess diagnostics.
        'band_coverage' : dict
            Geometric band diagnostic: how many observations fall within
            the (4/pi)^0.25 factor band around the base curve.
        'field_geometry' : dict
            Topologically predicted field geometry derived from the
            fitted mass model. Contains 'throat_radius_kpc' (R_t),
            'envelope_radius_kpc' (R_env), and 'yN_at_throat'
            (verification value, should equal 18/65). For dwarf galaxies
            where g_N never reaches a_0 * (4/13)(9/10), values are None.
    """

    # =================================================================
    # FIELD GEOMETRY: Derive R_t and R_env from the mass model
    #
    # The topological conditions y_N(R_t) = 18/65 and
    # y_N(R_env) = 36/1365 depend only on the baryonic mass profile
    # and a_0. No catalog galactic_radius is needed. We derive the
    # geometry from the published mass model here and use the result
    # for all downstream sigma stage constructions (Zero-SPARC).
    # After Stage 3 refines the masses, we re-derive.
    # =================================================================
    a0_eff = A0 * accel_ratio
    _b = mass_model.get("bulge", {})
    _d = mass_model.get("disk", {})
    _g = mass_model.get("gas", {})

    def _field_geometry_from(Mb, ab, Md, Rd, Mg, Rg):
        return solve_field_geometry(Mb, ab, Md, Rd, Mg, Rg, a0_eff)

    init_geom = _field_geometry_from(
        _b.get("M", 0), _b.get("a", 0),
        _d.get("M", 0), _d.get("Rd", 0),
        _g.get("M", 0), _g.get("Rd", 0))
    derived_R_env = init_geom["envelope_radius_kpc"]  # None for dwarfs

    # =================================================================
    # EARLY RETURN: Insufficient data for full inference
    # =================================================================
    # If the galaxy has fewer than 3 valid observations or the
    # topology does not yield a horizon (dwarf galaxies), we cannot
    # run the throughput/mass pipeline. Return the theoretical
    # throughput and the field geometry from the published mass model.
    # =================================================================
    fallback_ot = (auto_vortex_strength(mass_model, derived_R_env)
                   if derived_R_env else 0.0)

    if not observations or len(observations) < 3 or not derived_R_env:
        return {
            "mass_model": mass_model,
            "throughput": fallback_ot,
            "gfd_rms": None,
            "rms": None,
            "chi2_dof": None,
            "method": "theoretical",
            "field_geometry": init_geom,
        }

    # =================================================================
    # OBSERVATION PREPROCESSING
    #
    # Extract valid observations into parallel arrays for fast inner-loop
    # access. Invalid points (r <= 0 or v <= 0) are silently dropped.
    #
    # Weights are inverse-variance: w_j = 1 / err_j^2. This ensures
    # that high-precision measurements (e.g., Gaia DR3 with err = 1 km/s)
    # dominate the fit, while low-precision halo tracers (err = 30+ km/s)
    # contribute proportionally less.
    #
    # Error floor of 1.0 km/s prevents division-by-zero and avoids
    # giving unrealistically infinite weight to any single point.
    # =================================================================
    obs_r = []
    obs_v = []
    obs_w = []
    for o in observations:
        r = o.get('r', 0)
        v = o.get('v', 0)
        if r > 0 and v > 0:
            obs_r.append(float(r))
            obs_v.append(float(v))
            err = max(float(o.get('err', 5.0)), 1.0)
            obs_w.append(1.0 / (err * err))
    n_obs = len(obs_r)
    if n_obs < 3:
        return fallback

    # =================================================================
    # RADIAL GRID CONSTRUCTION
    #
    # Build a uniformly spaced radial grid from 0 to grid_max (kpc).
    # The grid extends 5% beyond the outermost observation to avoid
    # boundary effects in interpolation. Pre-compute the grid in
    # meters for the Newtonian acceleration calculation.
    # =================================================================
    max_obs_r = max(obs_r)
    # Grid must extend to at least R_env so the sigma stage's inner
    # vortex reflection has full outer delta data for mirror mapping.
    grid_max = max(max_radius, max_obs_r * 1.05,
                   derived_R_env * 1.05 if derived_R_env else 0)
    # Scale point count to maintain per-kpc density when the grid
    # extends beyond the requested max_radius.
    eff_n = num_points
    if grid_max > max_radius and max_radius > 0:
        scale = grid_max / max_radius
        eff_n = min(int(num_points * scale), 500)
    radii = [(grid_max / eff_n) * (i + 1)
             for i in range(eff_n)]
    n = len(radii)
    r_m = [r * KPC_TO_M for r in radii]
    # a0_eff already computed above (before early return check)

    # =================================================================
    # MASS MODEL EXTRACTION
    #
    # Unpack the three-component mass model. Each component is optional:
    # a galaxy may lack a bulge (dwarf irregulars), gas data (early-type
    # spirals), or in rare cases disk data. Boolean flags track which
    # components are present to skip unnecessary computation in the
    # inner loops.
    # =================================================================
    bulge = mass_model.get("bulge") or {}
    disk = mass_model.get("disk") or {}
    gas = mass_model.get("gas") or {}

    has_bulge = bulge.get("M", 0) > 0 and bulge.get("a", 0) > 0
    has_disk = disk.get("M", 0) > 0 and disk.get("Rd", 0) > 0
    has_gas = gas.get("M", 0) > 0 and gas.get("Rd", 0) > 0

    # =================================================================
    # ENCLOSED MASS HELPERS
    #
    # Inline for performance. These are called O(n_obs * n_grid) times
    # in the inner loops and must be fast. No external calls except
    # aqual_solve_x (the covariant field equation solver).
    #
    # _enc_bulge: Hernquist (1990) spherical profile.
    #   M_enc(r) = M * r^2 / (r + a)^2
    #   Produces a density cusp at the center and falls as r^-4 at large r.
    #
    # _enc_disk: Exponential disk (Freeman 1970) enclosed mass.
    #   M_enc(r) = M * [1 - (1 + r/Rd) * exp(-r/Rd)]
    #   Same functional form is used for both stellar and gas disks,
    #   with different scale lengths.
    #
    # _gfd_vel: GFD circular velocity from enclosed mass.
    #   Solves x^2/(1+x) = g_N/a0 and returns v = sqrt(a0 * x * r).
    #   This is the core GFD field equation (zero free parameters).
    # =================================================================
    def _enc_bulge(r, M, a):
        return M * r * r / ((r + a) * (r + a))

    def _enc_disk(r, M, Rd):
        x = r / Rd
        return M * (1.0 - (1.0 + x) * math.exp(-x))

    def _gfd_vel(enc, r_meters):
        if enc <= 0 or r_meters <= 0:
            return 0.0
        yN = G * enc * M_SUN / (r_meters * r_meters) / a0_eff
        x = aqual_solve_x(yN)
        return math.sqrt(a0_eff * x * r_meters) / 1000.0

    def _enc_total(r, Mb, ab, Md, Rd, Mg, Rg):
        enc = 0.0
        if has_bulge and Mb > 0 and ab > 0:
            enc += _enc_bulge(r, Mb, ab)
        if has_disk and Md > 0 and Rd > 0:
            enc += _enc_disk(r, Md, Rd)
        if has_gas and Mg > 0 and Rg > 0:
            enc += _enc_disk(r, Mg, Rg)
        return enc

    # =================================================================
    # STAGE 1: MASS BASELINE
    #
    # Accept the published photometric mass model without modification.
    #
    # Design rationale: The published masses are independently measured
    # (photometry, stellar population synthesis, 21-cm radio). Allowing
    # a kinematic optimizer to adjust individual components introduces
    # the bulge-disk degeneracy problem: at the radii where rotation
    # data exists, the curve is sensitive to total enclosed mass but
    # nearly insensitive to the partition among components. An optimizer
    # will exploit this freedom, destroying the physically meaningful
    # partition (see INFERENCE_PIPELINE.md, "The Degeneracy Problem").
    #
    # The total baryonic mass is well-constrained by kinematics (the
    # Tully-Fisher relation guarantees this). The partition is determined
    # by field closure topology (Stage 3), not by kinematic fitting.
    # =================================================================
    best_Mb = bulge.get("M", 0.0)
    best_ab = bulge.get("a", 0.1)
    best_Md = disk.get("M", 0.0)
    best_Rd = disk.get("Rd", 1.0)
    best_Mg = gas.get("M", 0.0)
    best_Rg = gas.get("Rd", 1.0)

    # -----------------------------------------------------------------
    # Weighted chi-squared function for the GFD base curve.
    # Used in the gene report (diagnostic) and could be used for
    # future quality gates. Interpolates the predicted velocity at
    # each observation radius by linear interpolation on the radial grid.
    # -----------------------------------------------------------------
    def gfd_chi2(Mb, ab, Md, Rd, Mg, Rg):
        chi2 = 0.0
        for j in range(n_obs):
            rj = obs_r[j]
            v_pred = 0.0
            for i in range(n - 1):
                if radii[i] <= rj <= radii[i + 1]:
                    enc_lo = _enc_total(radii[i], Mb, ab, Md, Rd, Mg, Rg)
                    enc_hi = _enc_total(radii[i + 1], Mb, ab, Md, Rd, Mg, Rg)
                    v_lo = _gfd_vel(enc_lo, r_m[i])
                    v_hi = _gfd_vel(enc_hi, r_m[i + 1])
                    dr = radii[i + 1] - radii[i]
                    f = (rj - radii[i]) / dr if dr > 0 else 0.0
                    v_pred = v_lo + f * (v_hi - v_lo)
                    break
            delta = obs_v[j] - v_pred
            chi2 += obs_w[j] * delta * delta
        return chi2

    # Build initial mass model output (may be refined by Stage 3)
    opt_model = {}
    if has_bulge:
        opt_model["bulge"] = {"M": round(best_Mb, 2), "a": round(best_ab, 2)}
    elif mass_model.get("bulge"):
        opt_model["bulge"] = dict(mass_model["bulge"])
    if has_disk:
        opt_model["disk"] = {"M": round(best_Md, 2), "Rd": round(best_Rd, 2)}
    elif mass_model.get("disk"):
        opt_model["disk"] = dict(mass_model["disk"])
    if has_gas:
        opt_model["gas"] = {"M": round(best_Mg, 2), "Rd": round(best_Rg, 2)}
    elif mass_model.get("gas"):
        opt_model["gas"] = dict(mass_model["gas"])

    # -----------------------------------------------------------------
    # GFD base curve RMS against observations.
    # This measures how well the published mass model predicts the
    # rotation curve *before* the structural correction (sigma) is
    # applied. A high base RMS is expected; the structural correction
    # in Stage 2 accounts for the residual.
    # -----------------------------------------------------------------
    gfd_ss = 0.0
    for j in range(n_obs):
        rj = obs_r[j]
        v_pred = 0.0
        for i in range(n - 1):
            if radii[i] <= rj <= radii[i + 1]:
                enc_lo = _enc_total(
                    radii[i], best_Mb, best_ab, best_Md, best_Rd,
                    best_Mg, best_Rg)
                enc_hi = _enc_total(
                    radii[i + 1], best_Mb, best_ab, best_Md, best_Rd,
                    best_Mg, best_Rg)
                v_lo = _gfd_vel(enc_lo, r_m[i])
                v_hi = _gfd_vel(enc_hi, r_m[i + 1])
                dr = radii[i + 1] - radii[i]
                f = (rj - radii[i]) / dr if dr > 0 else 0.0
                v_pred = v_lo + f * (v_hi - v_lo)
                break
        gfd_ss += (obs_v[j] - v_pred) ** 2
    gfd_rms = math.sqrt(gfd_ss / n_obs)

    # =================================================================
    # STAGE 2: ORIGIN THROUGHPUT OPTIMIZATION
    #
    # The Origin Throughput is a scalar modulator that controls the
    # amplitude of the structural correction (the "sigma" curve) around
    # the Field Origin at R_t = 0.30 * R_env.
    #
    # The search minimizes the weighted chi-squared between the
    # GFD-sigma prediction and observed velocities. This is a clean 1D
    # optimization problem because the throughput parameter enters the
    # sigma curve linearly through the effective gas factor:
    #   f_eff = (1 + f_gas) / 2 * throughput
    #
    # Two-pass grid search:
    #   Pass 1 (coarse): 61 evaluations at steps of 0.1 over [-3, +3]
    #   Pass 2 (fine):   21 evaluations at steps of 0.01 around the
    #                    coarse optimum
    #
    # Total: ~82 GFD-sigma evaluations. At ~0.5 ms each (500 grid
    # points), this completes in ~40 ms.
    #
    # GEOMETRIC_FACTOR = (4/pi)^0.25 defines the diagnostic band around
    # the base curve. This is NOT used for optimization; it is a
    # diagnostic to report how tightly the sigma curve follows the base.
    # =================================================================
    GEOMETRIC_FACTOR = (4.0 / math.pi) ** 0.25

    m_stellar_opt = best_Mb + best_Md
    m_total_opt = best_Mb + best_Md + best_Mg
    f_gas_opt = best_Mg / m_total_opt if m_total_opt > 0 else 0.0

    # Pre-compute enclosed mass profile on the radial grid
    opt_enc = [_enc_total(radii[i], best_Mb, best_ab, best_Md, best_Rd,
                          best_Mg, best_Rg) for i in range(n)]

    # Pre-compute GFD base velocity at each observation radius
    # (used for geometric band diagnostic)
    gfd_v_at_obs = []
    for j in range(n_obs):
        rj = obs_r[j]
        v_gfd = 0.0
        for i in range(n - 1):
            if radii[i] <= rj <= radii[i + 1]:
                enc_lo = _enc_total(
                    radii[i], best_Mb, best_ab, best_Md, best_Rd,
                    best_Mg, best_Rg)
                enc_hi = _enc_total(
                    radii[i + 1], best_Mb, best_ab, best_Md, best_Rd,
                    best_Mg, best_Rg)
                v_lo = _gfd_vel(enc_lo, r_m[i])
                v_hi = _gfd_vel(enc_hi, r_m[i + 1])
                dr = radii[i + 1] - radii[i]
                f = (rj - radii[i]) / dr if dr > 0 else 0.0
                v_gfd = v_lo + f * (v_hi - v_lo)
                break
        gfd_v_at_obs.append(v_gfd)

    obs_err = [max(o.get('err', 5.0), 1.0) for o in observations
               if o.get('r', 0) > 0 and o.get('v', 0) > 0]

    def _make_sigma_stage(throughput):
        """Construct a GFD-sigma stage for a given throughput value."""
        return GfdSymmetricStage(
            name="_opt", equation_label="",
            parameters={
                "accel_ratio": accel_ratio,
                "galactic_radius_kpc": derived_R_env,
                "m_stellar": m_stellar_opt,
                "f_gas": f_gas_opt,
                "vortex_strength": throughput,
            },
        )

    def band_coverage(throughput):
        """Diagnostic: count observations within the geometric band."""
        stage = _make_sigma_stage(throughput)
        result = stage.process(radii, opt_enc)
        v_sigma = result.series

        hits = 0
        overlap_score = 0.0
        band_ok = True

        for j in range(n_obs):
            v_s = GfdSymmetricStage._interp(radii, v_sigma, obs_r[j])
            v_base = gfd_v_at_obs[j]

            if v_base > 0:
                v_upper = v_base * GEOMETRIC_FACTOR
                v_lower = v_base / GEOMETRIC_FACTOR
                if v_s < v_lower or v_s > v_upper:
                    band_ok = False

            v_obs = obs_v[j]
            err = obs_err[j]
            if abs(v_obs - v_s) <= err:
                hits += 1
                overlap_score += 1.0 - abs(v_obs - v_s) / err

        return hits, overlap_score, band_ok

    def _sigma_chi2_at(throughput):
        """Weighted chi-squared of GFD-sigma vs observations."""
        stage = _make_sigma_stage(throughput)
        result = stage.process(radii, opt_enc)
        v_sigma = result.series
        c2 = 0.0
        for j in range(n_obs):
            v_s = GfdSymmetricStage._interp(radii, v_sigma, obs_r[j])
            delta = obs_v[j] - v_s
            c2 += obs_w[j] * delta * delta
        return c2

    # --- Throughput search bounds ---
    TP_MIN, TP_MAX = -3.0, 3.0
    best_tp = 0.0
    best_hits = -1
    best_overlap = -1.0
    best_in_band = False

    # --- Pass 1: Coarse grid (steps of 0.1) ---
    best_chi2_tp = 1e30
    for ti in range(-30, 31):
        t = ti * 0.1
        c2 = _sigma_chi2_at(t)
        hits, overlap, in_band = band_coverage(t)
        if c2 < best_chi2_tp:
            best_chi2_tp = c2
            best_tp = t
            best_hits = hits
            best_overlap = overlap
            best_in_band = in_band

    # --- Pass 2: Fine grid (steps of 0.01 around coarse optimum) ---
    coarse_tp = best_tp
    for ti in range(-10, 11):
        t = coarse_tp + ti * 0.01
        if t < TP_MIN or t > TP_MAX:
            continue
        c2 = _sigma_chi2_at(t)
        hits, overlap, in_band = band_coverage(t)
        if c2 < best_chi2_tp:
            best_chi2_tp = c2
            best_tp = t
            best_hits = hits
            best_overlap = overlap
            best_in_band = in_band

    # =================================================================
    # DIAGNOSTICS: Sigma curve quality metrics
    #
    # Compute RMS and chi-squared for the best-fit sigma curve.
    # These are reported to the UI for researcher evaluation.
    # =================================================================
    diag_stage = _make_sigma_stage(best_tp)
    diag_result = diag_stage.process(radii, opt_enc)
    diag_v = diag_result.series
    ss = 0.0
    for j in range(n_obs):
        v_pred = GfdSymmetricStage._interp(radii, diag_v, obs_r[j])
        ss += (obs_v[j] - v_pred) ** 2
    rms = math.sqrt(ss / n_obs)
    dof = max(n_obs - 2, 1)

    sigma_chi2_val = 0.0
    for j in range(n_obs):
        v_pred = GfdSymmetricStage._interp(radii, diag_v, obs_r[j])
        delta = obs_v[j] - v_pred
        sigma_chi2_val += obs_w[j] * delta * delta

    # =================================================================
    # STAGE 3: MASS DECOMPOSITION FROM SIGMA CURVE
    #
    # Iteratively refine the stellar mass using the sigma curve's
    # structural acceleration field to back-calculate the enclosed
    # baryonic mass at each observation point.
    #
    # Algorithm:
    #   1. From the sigma curve, extract the structural acceleration
    #      g_struct(r) at each observation radius.
    #   2. Subtract g_struct from the total observed acceleration
    #      (v_obs^2 / r) to isolate the DTG (base) acceleration.
    #   3. Invert the GFD field equation to recover the enclosed
    #      Newtonian mass from the DTG acceleration.
    #   4. Subtract the gas contribution (held fixed from 21-cm data)
    #      to get the enclosed stellar mass.
    #   5. Average the outer half of the observation-derived stellar
    #      masses (where the curve is most sensitive to total mass).
    #   6. Split stellar mass into bulge and disk using the published
    #      photometric B/T ratio (preserved from photometry).
    #   7. Recompute the enclosed mass profile, re-optimize throughput,
    #      and check convergence.
    #
    # Convergence criterion: relative change in total stellar mass
    # between iterations < 1%. Maximum iterations: 5.
    #
    # Gas mass is NEVER adjusted. It is the best-constrained component
    # (21-cm HI surveys + 1.33 He correction) and serves as an anchor.
    # =================================================================
    if has_bulge and has_disk:
        # Published bulge-to-total stellar ratio (photometric anchor)
        pub_bt = bulge["M"] / (bulge["M"] + disk["M"]) if (
            bulge["M"] + disk["M"] > 0) else 0.0

        MAX_ITER = 5
        CONV_TOL = 0.01
        prev_stellar = best_Mb + best_Md

        for _iter in range(MAX_ITER):
            # Extract structural acceleration grid from sigma curve
            g_struct_grid = diag_result.intermediates.get(
                "g_struct", [0.0] * n)

            # Back-calculate stellar mass at each observation radius
            obs_m_stellar = []
            for j in range(n_obs):
                rj = obs_r[j]
                rj_m = rj * KPC_TO_M
                vj_m = obs_v[j] * 1000.0

                if rj_m <= 0 or vj_m <= 0:
                    obs_m_stellar.append(0.0)
                    continue

                # Total observed acceleration: g_obs = v^2 / r
                g_total = vj_m * vj_m / rj_m

                # Structural contribution (interpolated from sigma grid)
                g_s = GfdSymmetricStage._interp(
                    radii, g_struct_grid, rj)

                # DTG (base) acceleration: what remains after removing
                # the structural correction
                g_dtg = g_total - g_s

                if g_dtg <= 0:
                    obs_m_stellar.append(0.0)
                    continue

                # Invert GFD field equation: from g_dtg, recover y_N
                # g_dtg = a0 * x, where x^2/(1+x) = y_N
                # -> y_N = x^2 / (1+x)
                # -> M_enc = y_N * r^2 * a0 / G
                x_dtg = g_dtg / a0_eff
                y_N = x_dtg * x_dtg / (1.0 + x_dtg)
                M_enc = y_N * rj_m * rj_m * a0_eff / G / M_SUN

                # Subtract gas to isolate stellar mass
                gas_enc = _enc_disk(rj, best_Mg, best_Rg)
                m_star = max(0.0, M_enc - gas_enc)
                obs_m_stellar.append(m_star)

            # Use the upper half of mass estimates (outer radii
            # give the most reliable total-mass constraint)
            half = max(1, n_obs // 2)
            outer_masses = sorted(obs_m_stellar)[-half:]
            outer_masses = [m for m in outer_masses if m > 0]
            if outer_masses:
                m_stellar_kin = sum(outer_masses) / len(outer_masses)
            else:
                m_stellar_kin = prev_stellar

            # Split stellar mass using photometric B/T ratio
            best_Mb = pub_bt * m_stellar_kin
            best_Md = (1.0 - pub_bt) * m_stellar_kin

            # Recompute enclosed mass profile with updated stellar masses
            opt_enc = [_enc_total(radii[i], best_Mb, best_ab,
                                  best_Md, best_Rd,
                                  best_Mg, best_Rg)
                       for i in range(n)]

            # Update sigma curve parameters
            m_stellar_opt = best_Mb + best_Md
            m_total_opt = m_stellar_opt + best_Mg
            f_gas_opt = (best_Mg / m_total_opt
                         if m_total_opt > 0 else 0.0)

            # Re-derive field geometry from the refined mass model so
            # the sigma stage uses the updated R_env (Zero-SPARC).
            refined_geom = _field_geometry_from(
                best_Mb, best_ab, best_Md, best_Rd, best_Mg, best_Rg)
            if refined_geom["envelope_radius_kpc"]:
                derived_R_env = refined_geom["envelope_radius_kpc"]

            # Re-optimize throughput with the updated mass model
            def _make_sigma_iter(throughput):
                return GfdSymmetricStage(
                    name="_opt", equation_label="",
                    parameters={
                        "accel_ratio": accel_ratio,
                        "galactic_radius_kpc": derived_R_env,
                        "m_stellar": m_stellar_opt,
                        "f_gas": f_gas_opt,
                        "vortex_strength": throughput,
                    },
                )

            # Coarse throughput search
            best_chi2_iter = 1e30
            for ti in range(-30, 31):
                t = ti * 0.1
                stg = _make_sigma_iter(t)
                res = stg.process(radii, opt_enc)
                c2 = 0.0
                for j in range(n_obs):
                    v_s = GfdSymmetricStage._interp(
                        radii, res.series, obs_r[j])
                    delta = obs_v[j] - v_s
                    c2 += obs_w[j] * delta * delta
                if c2 < best_chi2_iter:
                    best_chi2_iter = c2
                    best_tp = t

            # Fine throughput search
            coarse = best_tp
            for ti in range(-10, 11):
                t = coarse + ti * 0.01
                if t < -3.0 or t > 3.0:
                    continue
                stg = _make_sigma_iter(t)
                res = stg.process(radii, opt_enc)
                c2 = 0.0
                for j in range(n_obs):
                    v_s = GfdSymmetricStage._interp(
                        radii, res.series, obs_r[j])
                    delta = obs_v[j] - v_s
                    c2 += obs_w[j] * delta * delta
                if c2 < best_chi2_iter:
                    best_chi2_iter = c2
                    best_tp = t

            # Update sigma stage factory and recompute diagnostics
            _make_sigma_stage = _make_sigma_iter
            diag_stage = _make_sigma_stage(best_tp)
            diag_result = diag_stage.process(radii, opt_enc)
            diag_v = diag_result.series

            # Check convergence: has the total stellar mass stabilized?
            new_stellar = best_Mb + best_Md
            rel_change = (abs(new_stellar - prev_stellar)
                          / max(prev_stellar, 1.0))
            prev_stellar = new_stellar
            if rel_change < CONV_TOL:
                break

        # Update output mass model with decomposed values
        if has_bulge:
            opt_model["bulge"] = {"M": round(best_Mb, 2),
                                  "a": round(best_ab, 2)}
        if has_disk:
            opt_model["disk"] = {"M": round(best_Md, 2),
                                 "Rd": round(best_Rd, 2)}

        # Recompute RMS and chi-squared with final sigma curve
        ss = 0.0
        for j in range(n_obs):
            v_pred = GfdSymmetricStage._interp(
                radii, diag_v, obs_r[j])
            ss += (obs_v[j] - v_pred) ** 2
        rms = math.sqrt(ss / n_obs)

        sigma_chi2_val = 0.0
        for j in range(n_obs):
            v_pred = GfdSymmetricStage._interp(
                radii, diag_v, obs_r[j])
            delta = obs_v[j] - v_pred
            sigma_chi2_val += obs_w[j] * delta * delta

    # =================================================================
    # GENE REPORT: Per-parameter diagnostic comparison
    #
    # Compares each fitted parameter to its published (photometric)
    # value and reports whether the deviation exceeds the observational
    # uncertainty (sigma) for that parameter.
    #
    # Tier system:
    #   Tier 3 (highest): Mass parameters (Mb, Md) that directly affect
    #     the rotation curve and are most important for physics.
    #   Tier 2: Bulge scale (ab), which has moderate kinematic impact.
    #   Tier 1: Scale lengths and gas mass, which are better constrained
    #     by photometry than by kinematics.
    #
    # Sigma values represent the observational uncertainty (in dex for
    # masses, fractional for scale lengths):
    #   Mb: 0.30 dex (~factor of 2 range from different methods)
    #   Md: 0.30 dex (stellar mass-to-light ratio uncertainty)
    #   Mg: 0.10 dex (21-cm HI is the best-constrained component)
    #   Rd: 0.15 (fractional, ~15% from surface brightness fitting)
    #   ab: 0.30 (fractional, bulge geometry is poorly constrained)
    #   Rg: 0.20 (fractional, HI extent varies with survey depth)
    # =================================================================
    _label_map = {
        "Mb": "Bulge Mass", "ab": "Bulge Scale",
        "Md": "Disk Mass",  "Rd": "Disk Scale",
        "Mg": "Gas Mass",   "Rg": "Gas Scale",
    }
    _tier_map = {
        "Mb": 3, "ab": 2, "Md": 3, "Rd": 1, "Mg": 1, "Rg": 1,
    }

    _pub_vals = {}
    _fit_vals = {}
    if has_bulge:
        _pub_vals["Mb"] = bulge["M"]
        _pub_vals["ab"] = bulge["a"]
        _fit_vals["Mb"] = best_Mb
        _fit_vals["ab"] = best_ab
    if has_disk:
        _pub_vals["Md"] = disk["M"]
        _pub_vals["Rd"] = disk["Rd"]
        _fit_vals["Md"] = best_Md
        _fit_vals["Rd"] = best_Rd
    if has_gas:
        _pub_vals["Mg"] = gas["M"]
        _pub_vals["Rg"] = gas["Rd"]
        _fit_vals["Mg"] = best_Mg
        _fit_vals["Rg"] = best_Rg

    _obs_sigma = {
        "Mb": 0.30, "ab": 0.30, "Md": 0.30, "Rd": 0.15,
        "Mg": 0.10, "Rg": 0.20,
    }

    best_raw = gfd_chi2(best_Mb, best_ab, best_Md, best_Rd,
                        best_Mg, best_Rg)
    gene_report = []
    for label in _pub_vals:
        phys_init = _pub_vals[label]
        phys_val = _fit_vals[label]
        ratio = phys_val / phys_init if phys_init > 0 else 0.0

        is_mass = label in ("Mb", "Md", "Mg")
        if is_mass and phys_init > 0 and phys_val > 0:
            diff_dex = abs(math.log10(phys_val) - math.log10(phys_init))
            sigma = _obs_sigma.get(label, 0.3)
            sigma_excess = max(0.0, diff_dex - sigma) / sigma if sigma > 0 else 0.0
            within = diff_dex <= sigma
        else:
            frac_diff = abs(phys_val - phys_init) / abs(phys_init) if phys_init != 0 else 0.0
            sigma = _obs_sigma.get(label, 0.3)
            sigma_excess = max(0.0, frac_diff - sigma) / sigma if sigma > 0 else 0.0
            within = frac_diff <= sigma

        entry = {
            "gene": label,
            "name": _label_map.get(label, label),
            "tier": _tier_map.get(label, 0),
            "published": round(phys_init, 4),
            "fitted": round(phys_val, 4),
            "ratio": round(ratio, 3),
            "within_sigma": bool(within),
            "sigma_excess": round(sigma_excess, 2),
            "chi2_bought": 0.0,
            "chi2_per_obs": 0.0,
        }
        gene_report.append(entry)

    # =================================================================
    # FIELD GEOMETRY: Derive R_t and R_env from the fitted mass model
    #
    # Now that Stage 3 has refined the baryonic mass decomposition, we
    # can compute the field geometry purely from the mass distribution.
    # The throat condition y_N = (4/13)(9/10) is evaluated against the
    # FITTED masses (not published), because these represent the best
    # estimate of the true baryonic content after observation correction.
    #
    # This replaces the need for an externally supplied galactic_radius.
    # The returned values are the topologically predicted R_t and R_env,
    # derived from the three mass components and the fundamental constants.
    # =================================================================
    field_geometry = solve_field_geometry(
        best_Mb, best_ab, best_Md, best_Rd, best_Mg, best_Rg, a0_eff)

    # =================================================================
    # RETURN: Complete inference result
    # =================================================================
    return {
        "mass_model": opt_model,
        "throughput": round(best_tp, 2),
        "gfd_rms": round(gfd_rms, 2),
        "rms": round(rms, 2),
        "chi2_dof": round(sigma_chi2_val / dof, 2),
        "method": "inference",
        "gene_report": gene_report,
        "band_coverage": {
            "geometric_factor": round(GEOMETRIC_FACTOR, 4),
            "obs_hits": best_hits,
            "obs_total": n_obs,
            "within_band": best_in_band,
        },
        "field_geometry": field_geometry,
    }
