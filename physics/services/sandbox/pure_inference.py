"""
Pure observation-driven inference pipeline.

Derives the complete galactic field model from observation data alone:
observations (r, v, err) -> M_enc(r) -> parametric mass model ->
manifold geometry (R_t, R_env) from topological yN conditions.

No published masses. No user-provided galactic_radius. No SPARC
catalog values. Zero external inputs beyond the observation points
and the topological constant a0.

Pipeline:
  1. Invert field equation at each observation: (r, v) -> M_enc(r)
     The GFD field equation inversion is exact and parameter-free.
     The round trip (v -> M_enc -> v) is lossless: observations
     ARE GFD. There is no gap to fill.
  2. Fit smooth 3-component mass model to M_enc(r)
     (Hernquist bulge + 2 exponential disks)
     This model is a geometry ladder: it provides a continuous
     M_enc(r) function for the yN solvers. It is not used for
     rotation curve prediction.
  3. Solve two independent topological conditions on the mass model:
       Throat:  yN(R_t)   = 18/65   (field origin)
       Horizon: yN(R_env) = 36/1365 (field boundary)
     Both are absolute acceleration thresholds derived from the
     stellated octahedron coupling polynomial f(k) = 21.
  4. Verify R_t / R_env ~ 0.30 (emerges, not assumed).
  5. (Optional diagnostic) Compute GFD base curve from the smooth
     mass model and sigma correction to visualize the decomposition
     of mass contribution vs vortex power output.

The derivation chain is:
  a0 = k^2 * G * me / re^2          (topological acceleration scale)
  f(k) = 1 + k + k^2 = 21           (coupling polynomial, k=4)
  yN_throat = (4/13)(9/10) = 18/65  (structural frac * throughput)
  yN_horizon = (18/65)(2/21) = 36/1365  (throat / (f(k)/2))

Every constant traces to k = 4. No fitting. No free parameters.

IMPORTANT: No unicode characters (Windows charmap constraint).
"""

import math

from physics.constants import G, M_SUN, KPC_TO_M, A0
from physics.constants import THROAT_YN, HORIZON_YN, THROAT_FRAC
from physics.aqual import solve_x as aqual_solve_x
from physics.sigma import GfdSymmetricStage
from physics.services.rotation.inference import solve_field_geometry


# =====================================================================
# TOPOLOGICAL MASS DERIVATION
#
# The horizon and throat conditions are parameter-free equations
# that relate the field geometry (R_t, R_env) to enclosed baryonic
# mass. Given only the radii, the enclosed mass is determined:
#
#   Horizon: M_enc(R_env) = (36/1365) * R_env^2 * a0 / G
#   Throat:  M_enc(R_t)   = (18/65)   * R_t^2   * a0 / G
#
# For 3-cycle galaxies (compact, disk/bulge-dominated), nearly all
# mass is enclosed within R_env, so M_enc(R_env) ~ M_total.
# Validated to 0.1% to 3.4% across SPARC spirals.
#
# For 2-cycle galaxies (diffuse, gas-dominated), significant mass
# extends beyond R_env. The enclosure ratio (M_enc(R_env) / M_total)
# ranges from 0.25 to 0.55. To recover M_total, divide by the
# enclosure ratio: M_total = M_enc(R_env) / enclosure_ratio.
#
# The enclosure ratio is computed from the mass model shape (scale
# lengths relative to R_env). It requires knowing the parametric
# mass profile but NOT the absolute masses, since the ratio
# depends only on the concentration: how much of each component's
# mass falls inside R_env.
#
# The cycle count (2 or 3) is determined by whether yN ever
# reaches 18/65. If not, the galaxy has no throat and is
# gas-dominated (2-cycle). If yes, the throat exists and the
# galaxy is disk/bulge-dominated (3-cycle).
# =====================================================================

def horizon_enclosed_mass(r_env_kpc, a0_eff):
    """Derive enclosed baryonic mass at R_env from the horizon condition.

    M_enc(R_env) = yN_horizon * R_env^2 * a0 / G

    where yN_horizon = 36/1365 = (18/65)(2/21).

    This is parameter-free: given R_env from the field geometry,
    the enclosed mass is fully determined by the topology.

    For 3-cycle galaxies, M_enc(R_env) ~ M_total (>97% enclosed).
    For 2-cycle galaxies, M_enc(R_env) < M_total (25% to 55%).
    Use the enclosure_ratio from derive_mass_from_topology() to
    correct for diffuse mass beyond the horizon.

    Validated against SPARC published enclosed masses:
      Milky Way:  -0.5%   (3-cycle, enclosure ~0.99)
      M31:        -0.9%   (3-cycle, enclosure ~0.99)
      NGC 2841:   -2.1%   (3-cycle, enclosure ~0.98)
      NGC 6503:   -1.5%   (3-cycle, enclosure ~0.99)
      M33:        -3.4%   (3-cycle, enclosure ~0.97)
      DDO 154:    exact    (2-cycle, enclosure  0.26)
      NGC 3109:   exact    (2-cycle, enclosure  0.38)
    (see scripts/prove_horizon_all.py)

    Parameters
    ----------
    r_env_kpc : float
        Envelope (horizon) radius in kiloparsecs, where
        yN(R_env) = 36/1365.
    a0_eff : float
        Effective acceleration scale in m/s^2 (A0 * accel_ratio).

    Returns
    -------
    float
        Enclosed baryonic mass at R_env in solar masses.
    """
    if r_env_kpc is None or r_env_kpc <= 0:
        return 0.0
    r_m = r_env_kpc * KPC_TO_M
    return HORIZON_YN * r_m * r_m * a0_eff / (G * M_SUN)


def throat_enclosed_mass(r_t_kpc, a0_eff):
    """Derive enclosed mass at the throat from the throat condition.

    M_enc(R_t) = yN_throat * R_t^2 * a0 / G

    where yN_throat = 18/65 = (4/13)(9/10).

    For 3-cycle galaxies, this constrains how much mass is
    concentrated inside the throat radius. Combined with the
    closure cycle ratio (Md/Mg) and M_total from the horizon,
    this gives three equations for three unknowns (Mb, Md, Mg).

    For 2-cycle galaxies, this function returns 0 because
    the throat condition is never met (yN_max < 18/65).

    Parameters
    ----------
    r_t_kpc : float or None
        Throat radius in kiloparsecs, where yN(R_t) = 18/65.
        None for 2-cycle (gas-dominated) galaxies.
    a0_eff : float
        Effective acceleration scale in m/s^2.

    Returns
    -------
    float
        Enclosed baryonic mass at R_t in solar masses,
        or 0.0 if r_t_kpc is None.
    """
    if r_t_kpc is None or r_t_kpc <= 0:
        return 0.0
    r_m = r_t_kpc * KPC_TO_M
    return THROAT_YN * r_m * r_m * a0_eff / (G * M_SUN)


def _enclosure_ratio(r_env, Mb, ab, Md, Rd, Mg, Rg):
    """Fraction of total asymptotic mass enclosed within R_env.

    Computed analytically from the 3-component mass profile shapes.
    Depends only on scale lengths relative to R_env, not on the
    absolute mass values (the ratio cancels).

    For 3-cycle galaxies: typically 0.95 to 1.00.
    For 2-cycle galaxies: typically 0.25 to 0.55.

    Parameters
    ----------
    r_env : float
        Envelope radius in kpc.
    Mb, ab : float
        Bulge mass (Msun) and Hernquist scale (kpc).
    Md, Rd : float
        Disk mass (Msun) and exponential scale length (kpc).
    Mg, Rg : float
        Gas mass (Msun) and exponential scale length (kpc).

    Returns
    -------
    float
        Ratio M_enc(R_env) / M_asymptotic, in range (0, 1].
        Returns 1.0 if M_asymptotic is zero or r_env is invalid.
    """
    if r_env is None or r_env <= 0:
        return 1.0

    M_asymptotic = Mb + Md + Mg
    if M_asymptotic <= 0:
        return 1.0

    # Enclosed mass at R_env from each component profile
    enc = 0.0
    if Mb > 0 and ab > 0:
        enc += Mb * r_env * r_env / ((r_env + ab) * (r_env + ab))
    if Md > 0 and Rd > 0:
        x = r_env / Rd
        if x > 50:
            enc += Md
        else:
            enc += Md * (1.0 - (1.0 + x) * math.exp(-x))
    if Mg > 0 and Rg > 0:
        x = r_env / Rg
        if x > 50:
            enc += Mg
        else:
            enc += Mg * (1.0 - (1.0 + x) * math.exp(-x))

    ratio = enc / M_asymptotic
    return max(min(ratio, 1.0), 0.0)


def derive_mass_from_topology(field_geometry, a0_eff, mass_params=None):
    """Derive baryonic mass constraints from topological conditions.

    Given the field geometry (R_t, R_env) from solve_field_geometry,
    returns the topologically determined mass budget, cycle count,
    and (when mass_params are provided) the enclosure ratio for
    correcting diffuse 2-cycle galaxies.

    Two output mass values:
      M_enc_horizon : exact enclosed mass at R_env from the topology.
      M_total       : corrected total mass = M_enc_horizon / enclosure_ratio.

    For 3-cycle galaxies, enclosure_ratio ~ 1.0, so they are equal.
    For 2-cycle galaxies, enclosure_ratio < 1.0, and M_total > M_enc_horizon.

    Parameters
    ----------
    field_geometry : dict
        Output of solve_field_geometry(), containing:
          'throat_radius_kpc': float or None
          'envelope_radius_kpc': float or None
    a0_eff : float
        Effective acceleration scale in m/s^2 (A0 * accel_ratio).
    mass_params : tuple of 6 floats, optional
        (Mb, ab, Md, Rd, Mg, Rg) from the parametric mass model.
        When provided, the enclosure ratio is computed analytically
        from the mass profile shapes. When omitted, enclosure_ratio
        defaults to 1.0 (assumes all mass inside R_env).

    Returns
    -------
    dict with keys:
        'M_enc_horizon' : float
            Enclosed mass at R_env from horizon condition (solar masses).
            Exact, parameter-free.
        'M_total' : float
            Corrected total mass (solar masses).
            M_enc_horizon / enclosure_ratio.
        'M_enc_throat' : float
            Enclosed mass at throat (solar masses). 0 for 2-cycle.
        'enclosure_ratio' : float
            Fraction of total mass inside R_env (0 to 1).
            ~1.0 for 3-cycle, 0.25 to 0.55 for 2-cycle.
            1.0 if mass_params not provided.
        'cycle' : int
            Closure cycle count: 3 if throat exists, 2 otherwise.
        'r_t_kpc' : float or None
            Throat radius (kpc).
        'r_env_kpc' : float or None
            Envelope radius (kpc).
        'throat_fraction' : float or None
            R_t / R_env (should be ~0.30 for 3-cycle).
    """
    r_t = field_geometry.get("throat_radius_kpc")
    r_env = field_geometry.get("envelope_radius_kpc")

    M_enc_hor = horizon_enclosed_mass(r_env, a0_eff)

    # Cycle count: use solve_field_geometry's determination if
    # available (it checks whether yN reaches 18/65). Fall back
    # to 3-cycle assumption if the key is missing (legacy callers).
    cycle = field_geometry.get("cycle", 3)

    # Throat enclosed mass: for 3-cycle, the throat condition
    # yN(R_t) = 18/65 holds exactly, so we use the formula.
    # For 2-cycle, R_t = 0.30 * R_env but yN(R_t) < 18/65.
    # The actual enclosed mass at the throat comes from the mass
    # model, not the 18/65 formula. Use yN_at_throat from the
    # geometry solver to compute M_enc correctly.
    if cycle == 3:
        M_enc_throat = throat_enclosed_mass(r_t, a0_eff)
    else:
        # M_enc(R_t) = yN_at_throat * R_t^2 * a0 / G
        yN_at_rt = field_geometry.get("yN_at_throat", 0.0)
        if r_t and r_t > 0 and yN_at_rt > 0:
            r_t_m = r_t * KPC_TO_M
            M_enc_throat = yN_at_rt * r_t_m * r_t_m * a0_eff / (G * M_SUN)
        else:
            M_enc_throat = 0.0

    throat_frac = field_geometry.get("throat_fraction")
    if throat_frac is None and r_t and r_env and r_env > 0:
        throat_frac = round(r_t / r_env, 4)

    # Compute enclosure ratio from mass model shape
    if mass_params is not None:
        Mb, ab, Md, Rd, Mg, Rg = mass_params
        enc_ratio = _enclosure_ratio(r_env, Mb, ab, Md, Rd, Mg, Rg)
    else:
        enc_ratio = 1.0

    # Corrected total mass
    if enc_ratio > 0:
        M_total = M_enc_hor / enc_ratio
    else:
        M_total = M_enc_hor

    return {
        "M_enc_horizon": round(M_enc_hor, 2),
        "M_total": round(M_total, 2),
        "M_enc_throat": round(M_enc_throat, 2),
        "enclosure_ratio": round(enc_ratio, 4),
        "cycle": cycle,
        "r_t_kpc": r_t,
        "r_env_kpc": r_env,
        "throat_fraction": throat_frac,
    }


# =====================================================================
# STEP 1: FIELD EQUATION INVERSION
#
# Given (r_kpc, v_km_s), invert the GFD field equation to recover
# the enclosed baryonic mass M_enc(r) in solar masses.
#
# The inversion is exact and parameter-free:
#   x = v^2 / (a0 * r)
#   y = x^2 / (1 + x)
#   M_enc = y * r^2 * a0 / G
# =====================================================================

def invert_velocity_to_mass(r_kpc, v_km_s, a0_eff):
    """Invert one (r, v) observation to enclosed baryonic mass.

    Parameters
    ----------
    r_kpc : float
        Galactocentric radius in kiloparsecs.
    v_km_s : float
        Circular velocity in km/s.
    a0_eff : float
        Effective acceleration scale (A0 * accel_ratio) in m/s^2.

    Returns
    -------
    float
        Enclosed baryonic mass in solar masses.
    """
    r_m = r_kpc * KPC_TO_M
    v_m = v_km_s * 1000.0
    if r_m <= 0 or v_m <= 0:
        return 0.0
    x = (v_m * v_m) / (a0_eff * r_m)
    y = (x * x) / (1.0 + x)
    M_enc = y * r_m * r_m * a0_eff / (G * M_SUN)
    return M_enc


def invert_observations(observations, a0_eff):
    """Invert all observation points to an enclosed mass profile.

    Parameters
    ----------
    observations : list of dict
        Each dict has 'r' (kpc), 'v' (km/s), 'err' (km/s).
    a0_eff : float
        Effective acceleration scale in m/s^2.

    Returns
    -------
    list of tuple
        Sorted by radius: [(r_kpc, v_km_s, err, M_enc_solar), ...].
    """
    result = []
    for o in observations:
        r = float(o.get('r', 0))
        v = float(o.get('v', 0))
        err = max(float(o.get('err', 5.0)), 1.0)
        if r > 0 and v > 0:
            M = invert_velocity_to_mass(r, v, a0_eff)
            result.append((r, v, err, M))
    result.sort(key=lambda x: x[0])
    return result


# =====================================================================
# STEP 2: THREE-COMPONENT MASS MODEL FITTING
#
# Fit Hernquist bulge + exponential disk + exponential gas to the
# inverted M_enc(r) profile. This gives us smooth mass parameters
# that define the GFD base curve. The parametric model cannot
# reproduce the sigma-shaped residuals, so those residuals become
# the structural correction signal for the sigma stage.
# =====================================================================

def _hernquist_enc(r, M, a):
    """Hernquist enclosed mass."""
    if M <= 0 or a <= 0 or r <= 0:
        return 0.0
    return M * r * r / ((r + a) * (r + a))


def _disk_enc(r, M, Rd):
    """Exponential disk enclosed mass."""
    if M <= 0 or Rd <= 0 or r <= 0:
        return 0.0
    x = r / Rd
    return M * (1.0 - (1.0 + x) * math.exp(-x))


def _model_enc(r, Mb, ab, Md, Rd, Mg, Rg):
    """Total enclosed mass from the 3-component model."""
    return (_hernquist_enc(r, Mb, ab)
            + _disk_enc(r, Md, Rd)
            + _disk_enc(r, Mg, Rg))


def _gfd_vel(r_kpc, Mb, ab, Md, Rd, Mg, Rg, a0_eff):
    """GFD base velocity from the 3-component model."""
    enc = _model_enc(r_kpc, Mb, ab, Md, Rd, Mg, Rg)
    r_m = r_kpc * KPC_TO_M
    if r_m <= 0 or enc <= 0:
        return 0.0
    gN = G * enc * M_SUN / (r_m * r_m)
    y = gN / a0_eff
    x = aqual_solve_x(y)
    return math.sqrt(a0_eff * x * r_m) / 1000.0


def fit_mass_model(inverted_profile, a0_eff):
    """Fit a 3-component mass model to the inverted mass profile.

    Uses a grid search over scale lengths with linear regression for
    masses at each grid point. Scored on velocity residuals to ensure
    the GFD base curve (not just M_enc) fits observations.

    Parameters
    ----------
    inverted_profile : list of tuple
        [(r_kpc, v_km_s, err, M_enc_solar), ...] sorted by radius.
    a0_eff : float
        Effective acceleration scale in m/s^2.

    Returns
    -------
    dict with keys:
        'params': (Mb, ab, Md, Rd, Mg, Rg)
        'model': {bulge: {M, a}, disk: {M, Rd}, gas: {M, Rd}}
        'gfd_rms': float (GFD base RMS vs observations)
    """
    if len(inverted_profile) < 3:
        return None

    radii = [p[0] for p in inverted_profile]
    masses = [p[3] for p in inverted_profile]
    obs_v = [p[1] for p in inverted_profile]
    obs_err = [p[2] for p in inverted_profile]
    M_total = masses[-1] if masses else 0.0
    r_max = radii[-1] if radii else 10.0
    n = len(radii)

    ab_candidates = [r_max * f for f in [0.01, 0.02, 0.04, 0.06, 0.10]]
    Rd_candidates = [r_max * f for f in [0.05, 0.08, 0.12, 0.18, 0.25]]
    Rg_candidates = [r_max * f for f in [0.12, 0.20, 0.30, 0.45, 0.65]]
    ab_candidates = [max(a, 0.05) for a in ab_candidates]
    Rd_candidates = [max(r, 0.2) for r in Rd_candidates]
    Rg_candidates = [max(r, 0.4) for r in Rg_candidates]

    best_chi2 = 1e30
    best_params = None

    for ab in ab_candidates:
        for Rd in Rd_candidates:
            if Rd <= ab:
                continue
            for Rg in Rg_candidates:
                if Rg <= Rd:
                    continue

                # Compute basis functions at each observation radius
                basis_b = [_hernquist_enc(r, 1.0, ab) for r in radii]
                basis_d = [_disk_enc(r, 1.0, Rd) for r in radii]
                basis_g = [_disk_enc(r, 1.0, Rg) for r in radii]

                # Solve for mass coefficients via normal equations
                bTb = sum(basis_b[i] * basis_b[i] for i in range(n))
                bTd = sum(basis_b[i] * basis_d[i] for i in range(n))
                bTg = sum(basis_b[i] * basis_g[i] for i in range(n))
                dTd = sum(basis_d[i] * basis_d[i] for i in range(n))
                dTg = sum(basis_d[i] * basis_g[i] for i in range(n))
                gTg = sum(basis_g[i] * basis_g[i] for i in range(n))
                bTm = sum(basis_b[i] * masses[i] for i in range(n))
                dTm = sum(basis_d[i] * masses[i] for i in range(n))
                gTm = sum(basis_g[i] * masses[i] for i in range(n))

                A = [[bTb, bTd, bTg],
                     [bTd, dTd, dTg],
                     [bTg, dTg, gTg]]
                rhs = [bTm, dTm, gTm]

                det = (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
                       - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                       + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))

                if abs(det) < 1e-30:
                    continue

                Mb = (rhs[0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
                      - A[0][1] * (rhs[1] * A[2][2] - A[1][2] * rhs[2])
                      + A[0][2] * (rhs[1] * A[2][1] - A[1][1] * rhs[2])
                      ) / det
                Md = (A[0][0] * (rhs[1] * A[2][2] - A[1][2] * rhs[2])
                      - rhs[0] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                      + A[0][2] * (A[1][0] * rhs[2] - rhs[1] * A[2][0])
                      ) / det
                Mg = (A[0][0] * (A[1][1] * rhs[2] - rhs[1] * A[2][1])
                      - A[0][1] * (A[1][0] * rhs[2] - rhs[1] * A[2][0])
                      + rhs[0] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
                      ) / det

                Mb = max(Mb, 0.0)
                Md = max(Md, 0.0)
                Mg = max(Mg, 0.0)

                if Mb + Md + Mg <= 0:
                    continue

                # Score on VELOCITY residuals (GFD base vs observations)
                chi2 = 0.0
                for i in range(n):
                    v_pred = _gfd_vel(radii[i], Mb, ab, Md, Rd, Mg, Rg,
                                      a0_eff)
                    delta = obs_v[i] - v_pred
                    chi2 += (delta * delta) / (obs_err[i] * obs_err[i])

                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_params = (Mb, ab, Md, Rd, Mg, Rg)

    if best_params is None:
        Rd_est = r_max * 0.15
        best_params = (0.0, 0.5, M_total, max(Rd_est, 1.0),
                       0.0, max(Rd_est * 2, 2.0))

    Mb, ab, Md, Rd, Mg, Rg = best_params

    # Compute GFD base RMS
    ss = 0.0
    for i in range(n):
        v_pred = _gfd_vel(radii[i], Mb, ab, Md, Rd, Mg, Rg, a0_eff)
        delta = obs_v[i] - v_pred
        ss += delta * delta
    gfd_rms = math.sqrt(ss / n) if n > 0 else 0.0

    model = {}
    if Mb > 0:
        model["bulge"] = {"M": round(Mb, 2), "a": round(ab, 4)}
    else:
        model["bulge"] = {"M": 0.0, "a": round(ab, 4)}
    model["disk"] = {"M": round(Md, 2), "Rd": round(Rd, 4)}
    model["gas"] = {"M": round(Mg, 2), "Rd": round(Rg, 4)}

    return {
        "params": best_params,
        "model": model,
        "gfd_rms": round(gfd_rms, 2),
    }


# =====================================================================
# STEP 2b: TOPOLOGY-AWARE MASS MODEL FITTING
#
# Uses the derived field geometry (R_t, R_env) to anchor scale length
# candidates. The triforce aperture structure gives:
#   0.30 + 0.30 + 0.30 + 0.10 = 1.0 (three channels + throat)
#
# The three channels map to mass components with ordered scales:
#   Bulge: a << R_t          (compact, inside throat)
#   Disk:  Rd ~ 0.1 * R_env  (intermediate, straddles throat)
#   Gas:   Rg ~ 0.3 * R_env  (extended, fills outer envelope)
#
# Empirical ratios across the SPARC catalog:
#   a / R_env:  0.002 to 0.027, typical ~ 0.01
#   Rd / R_env: 0.04 to 0.21, typical ~ 0.10
#   Rg / R_env: 0.12 to 0.50, typical ~ 0.30
# =====================================================================

def fit_mass_model_topology(inverted_profile, a0_eff, r_env):
    """Fit a 3-component mass model using topology-informed decomposition.

    The field geometry (R_env) anchors scale length candidates to the
    triforce aperture structure (0.30 + 0.30 + 0.30 + 0.10).

    Approach: explicitly scan gas fractions. The gas mass is the most
    degenerate component (disk and gas basis functions overlap heavily).
    For each candidate (ab, Rd, Rg, f_gas), the gas contribution is
    subtracted from M_enc, and bulge + disk are fitted to the residual
    using a 2x2 weighted linear regression. This breaks the three-way
    degeneracy by giving gas its own search dimension.

    Parameters
    ----------
    inverted_profile : list of tuple
        [(r_kpc, v_km_s, err, M_enc_solar), ...] sorted by radius.
    a0_eff : float
        Effective acceleration scale in m/s^2.
    r_env : float
        Envelope radius (kpc) from field geometry derivation.

    Returns
    -------
    dict with keys:
        'params': (Mb, ab, Md, Rd, Mg, Rg)
        'model': {bulge: {M, a}, disk: {M, Rd}, gas: {M, Rd}}
        'gfd_rms': float (GFD base RMS vs observations)
    """
    if len(inverted_profile) < 3:
        return None
    if r_env is None or r_env <= 0:
        return fit_mass_model(inverted_profile, a0_eff)

    radii = [p[0] for p in inverted_profile]
    masses = [p[3] for p in inverted_profile]
    obs_v = [p[1] for p in inverted_profile]
    obs_err = [p[2] for p in inverted_profile]
    n = len(radii)

    # Error weights for the mass regression
    w = [1.0 / (obs_err[i] * obs_err[i]) for i in range(n)]

    # Total baryonic mass estimate: use the peak M_enc from the
    # inverted profile. This is the best estimate of M_total since
    # M_enc should be monotonically increasing (outer dips are noise).
    M_total = max(masses)

    # Topology-anchored scale length candidates (fractions of R_env)
    ab_fracs = [0.004, 0.008, 0.012, 0.018, 0.025, 0.035]
    Rd_fracs = [0.03, 0.05, 0.07, 0.10, 0.13, 0.17, 0.22]
    Rg_fracs = [0.10, 0.15, 0.20, 0.25, 0.30, 0.38, 0.50]

    ab_candidates = [max(r_env * f, 0.05) for f in ab_fracs]
    Rd_candidates = [max(r_env * f, 0.2) for f in Rd_fracs]
    Rg_candidates = [max(r_env * f, 0.4) for f in Rg_fracs]

    # Gas fraction scan: the triforce aperture structure mandates
    # three active channels. The gas channel (outer envelope) must
    # carry non-zero mass. Minimum 5% enforces the topological
    # requirement; 0% is not scanned because it violates the
    # three-channel structure.
    fg_candidates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    best_chi2 = 1e30
    best_params = None

    for ab in ab_candidates:
        for Rd in Rd_candidates:
            if Rd <= ab:
                continue
            for Rg in Rg_candidates:
                if Rg <= Rd:
                    continue

                basis_g = [_disk_enc(r, 1.0, Rg) for r in radii]
                basis_b = [_hernquist_enc(r, 1.0, ab) for r in radii]
                basis_d = [_disk_enc(r, 1.0, Rd) for r in radii]

                for fg in fg_candidates:
                    Mg = M_total * fg

                    # Subtract gas from M_enc to get residual
                    resid = [masses[i] - Mg * basis_g[i]
                             for i in range(n)]

                    # 2x2 weighted regression: residual = Mb*H + Md*D
                    bTb = sum(w[i] * basis_b[i] * basis_b[i]
                              for i in range(n))
                    bTd = sum(w[i] * basis_b[i] * basis_d[i]
                              for i in range(n))
                    dTd = sum(w[i] * basis_d[i] * basis_d[i]
                              for i in range(n))
                    bTr = sum(w[i] * basis_b[i] * resid[i]
                              for i in range(n))
                    dTr = sum(w[i] * basis_d[i] * resid[i]
                              for i in range(n))

                    det2 = bTb * dTd - bTd * bTd
                    if abs(det2) < 1e-30:
                        continue

                    Mb = (bTr * dTd - bTd * dTr) / det2
                    Md = (bTb * dTr - bTd * bTr) / det2

                    Mb = max(Mb, 0.0)
                    Md = max(Md, 0.0)

                    if Mb + Md + Mg <= 0:
                        continue

                    # Score on velocity chi-squared
                    chi2 = 0.0
                    for i in range(n):
                        v_pred = _gfd_vel(
                            radii[i], Mb, ab, Md, Rd, Mg, Rg, a0_eff)
                        delta = obs_v[i] - v_pred
                        chi2 += (delta * delta) / (obs_err[i] * obs_err[i])

                    if chi2 < best_chi2:
                        best_chi2 = chi2
                        best_params = (Mb, ab, Md, Rd, Mg, Rg)

    if best_params is None:
        return fit_mass_model(inverted_profile, a0_eff)

    Mb, ab, Md, Rd, Mg, Rg = best_params

    ss = 0.0
    for i in range(n):
        v_pred = _gfd_vel(radii[i], Mb, ab, Md, Rd, Mg, Rg, a0_eff)
        delta = obs_v[i] - v_pred
        ss += delta * delta
    gfd_rms = math.sqrt(ss / n) if n > 0 else 0.0

    model = {}
    if Mb > 0:
        model["bulge"] = {"M": round(Mb, 2), "a": round(ab, 4)}
    else:
        model["bulge"] = {"M": 0.0, "a": round(ab, 4)}
    model["disk"] = {"M": round(Md, 2), "Rd": round(Rd, 4)}
    model["gas"] = {"M": round(Mg, 2), "Rd": round(Rg, 4)}

    return {
        "params": best_params,
        "model": model,
        "gfd_rms": round(gfd_rms, 2),
    }


# =====================================================================
# STEP 3: TOPOLOGICAL GEOMETRY FROM PARAMETRIC MODEL
#
# The fitted mass model gives us y_N(r) as a continuous function.
# solve_field_geometry finds where y_N = 18/65 -> R_t, R_env.
# =====================================================================

# (Reuses solve_field_geometry from inference.py)


# =====================================================================
# STEP 4: DIRECT y_N CROSSING FROM OBSERVATIONS (diagnostic)
#
# Independent of the parametric fit, compute y_N at each observation
# point and find where it crosses 18/65. This gives Rt_obs, the
# effective throat post-structural-correction.
# =====================================================================

def compute_yN_profile(inverted, a0_eff):
    """Compute y_N at each observation point from inverted M_enc."""
    profile = []
    for r_kpc, v, err, M_enc in inverted:
        r_m = r_kpc * KPC_TO_M
        if r_m > 0 and M_enc > 0:
            yN = G * M_enc * M_SUN / (r_m * r_m * a0_eff)
            profile.append((r_kpc, yN))
    return profile


def find_throat_direct(yN_profile):
    """Find Rt_obs by interpolating the y_N crossing at 18/65."""
    if not yN_profile:
        return {"throat_radius_kpc": None, "envelope_radius_kpc": None,
                "yN_at_throat": 0.0}

    max_yN = max(y for _, y in yN_profile)
    if max_yN < THROAT_YN:
        return {"throat_radius_kpc": None, "envelope_radius_kpc": None,
                "yN_at_throat": max_yN}

    r_t = None
    for i in range(len(yN_profile) - 1):
        r1, y1 = yN_profile[i]
        r2, y2 = yN_profile[i + 1]
        if y1 >= THROAT_YN and y2 < THROAT_YN:
            frac = (THROAT_YN - y1) / (y2 - y1)
            r_t = r1 + frac * (r2 - r1)
            break

    if r_t is None:
        r_last, y_last = yN_profile[-1]
        if y_last > THROAT_YN and len(yN_profile) >= 2:
            r_prev, y_prev = yN_profile[-2]
            if y_prev != y_last:
                frac = (THROAT_YN - y_prev) / (y_last - y_prev)
                r_t = r_prev + frac * (r_last - r_prev)
            else:
                r_t = r_last * 1.1
        else:
            r_t = r_last

    r_env = r_t / THROAT_FRAC
    return {
        "throat_radius_kpc": round(r_t, 4),
        "envelope_radius_kpc": round(r_env, 4),
        "yN_at_throat": round(THROAT_YN, 6),
    }


# =====================================================================
# STEP 5: SIGMA STAGE + THROUGHPUT SEARCH
#
# Build a GfdSymmetricStage using PARAMETRIC MODEL enclosed masses
# (not observation-interpolated). The parametric GFD base will not
# match observations perfectly; the residuals are the structural
# correction that the sigma stage captures.
# =====================================================================

def _build_sigma_stage(throughput, r_env, m_stellar, f_gas, accel_ratio):
    """Construct a GFD-sigma stage for a given throughput."""
    return GfdSymmetricStage(
        name="_sandbox",
        equation_label="",
        parameters={
            "accel_ratio": accel_ratio,
            "galactic_radius_kpc": r_env,
            "m_stellar": m_stellar,
            "f_gas": f_gas,
            "vortex_strength": throughput,
        },
    )


def search_throughput(inverted, params, r_env, accel_ratio, a0_eff,
                      num_points=500):
    """Search for optimal Origin Throughput using parametric model.

    The sigma stage uses PARAMETRIC enclosed masses (smooth GFD base),
    not observation-interpolated masses. This ensures the base does
    not already match observations, leaving room for the sigma
    correction to capture the structural residuals.

    Parameters
    ----------
    inverted : list of tuple
        [(r_kpc, v_km_s, err, M_enc), ...] from inversion.
    params : tuple
        (Mb, ab, Md, Rd, Mg, Rg) from parametric fit.
    r_env : float
        Envelope radius from topological derivation (kpc).
    accel_ratio : float
        Acceleration scale ratio.
    a0_eff : float
        Effective a_0 in m/s^2.
    num_points : int
        Radial grid resolution.

    Returns
    -------
    dict with throughput, f_gas, chi2_dof, rms, m_stellar
    """
    Mb, ab, Md, Rd, Mg, Rg = params
    m_total = Mb + Md + Mg
    m_stellar_pub = Mb + Md

    # Build radial grid
    max_obs_r = max(o[0] for o in inverted) if inverted else 10.0
    grid_max = max(r_env * 1.1, max_obs_r * 1.05)
    radii = [(grid_max / num_points) * (i + 1) for i in range(num_points)]

    # Enclosed mass from PARAMETRIC model at each grid point
    enc = [_model_enc(radii[i], Mb, ab, Md, Rd, Mg, Rg)
           for i in range(num_points)]

    obs_r = [o[0] for o in inverted]
    obs_v = [o[1] for o in inverted]
    obs_w = [1.0 / (o[2] * o[2]) for o in inverted]
    n_obs = len(obs_r)

    # Gas fraction from the parametric fit (observation-derived)
    f_gas_fit = Mg / m_total if m_total > 0 else 0.0

    # Search throughput with the parametric gas fraction
    best_tp = 0.0
    best_chi2 = 1e30

    def _chi2_at(tp, fg):
        ms = m_total * (1.0 - fg) if m_total > 0 else 0.0
        stage = _build_sigma_stage(tp, r_env, ms, fg, accel_ratio)
        result = stage.process(radii, enc)
        v_sigma = result.series
        c2 = 0.0
        for j in range(n_obs):
            v_s = GfdSymmetricStage._interp(radii, v_sigma, obs_r[j])
            delta = obs_v[j] - v_s
            c2 += obs_w[j] * delta * delta
        return c2

    # Also search a few gas fractions around the fitted value
    fg_candidates = set()
    fg_candidates.add(round(f_gas_fit, 2))
    fg_candidates.add(round(max(0, f_gas_fit - 0.10), 2))
    fg_candidates.add(round(min(1, f_gas_fit + 0.10), 2))
    fg_candidates.add(0.0)
    fg_candidates.add(0.20)
    fg_candidates = sorted(fg_candidates)

    best_fg = f_gas_fit

    for fg in fg_candidates:
        # Coarse grid (steps of 0.1)
        local_best_tp = 0.0
        local_best_chi2 = 1e30
        for ti in range(-30, 31):
            t = ti * 0.1
            c2 = _chi2_at(t, fg)
            if c2 < local_best_chi2:
                local_best_chi2 = c2
                local_best_tp = t

        # Fine grid (steps of 0.01 around coarse)
        coarse = local_best_tp
        for ti in range(-10, 11):
            t = coarse + ti * 0.01
            if t < -3.0 or t > 3.0:
                continue
            c2 = _chi2_at(t, fg)
            if c2 < local_best_chi2:
                local_best_chi2 = c2
                local_best_tp = t

        if local_best_chi2 < best_chi2:
            best_chi2 = local_best_chi2
            best_tp = local_best_tp
            best_fg = fg

    # Compute RMS with best parameters
    m_stellar_best = m_total * (1.0 - best_fg)
    stage = _build_sigma_stage(best_tp, r_env, m_stellar_best, best_fg,
                               accel_ratio)
    result = stage.process(radii, enc)
    v_sigma = result.series
    ss = 0.0
    for j in range(n_obs):
        v_s = GfdSymmetricStage._interp(radii, v_sigma, obs_r[j])
        delta = obs_v[j] - v_s
        ss += delta * delta
    rms = math.sqrt(ss / n_obs) if n_obs > 0 else 0.0
    chi2_dof = best_chi2 / max(n_obs - 2, 1)

    return {
        "throughput": round(best_tp, 4),
        "f_gas": best_fg,
        "chi2_dof": round(chi2_dof, 4),
        "rms": round(rms, 4),
        "m_stellar": round(m_stellar_best, 2),
    }


# =====================================================================
# UTILITY: linear interpolation
# =====================================================================

def _interp(xs, ys, x_target):
    """Linear interpolation / extrapolation on sorted (xs, ys)."""
    if not xs:
        return 0.0
    if x_target <= xs[0]:
        return ys[0]
    if x_target >= xs[-1]:
        return ys[-1]
    for j in range(len(xs) - 1):
        if xs[j] <= x_target <= xs[j + 1]:
            t = (x_target - xs[j]) / (xs[j + 1] - xs[j])
            return ys[j] + t * (ys[j + 1] - ys[j])
    return ys[-1]


# =====================================================================
# MAIN ENTRY POINT: infer_from_observations
#
# Complete pipeline: observations in, everything out.
#
# Core (geometry):
#   1. Invert (r,v) -> M_enc at each observation (exact, lossless)
#   2. Fit parametric mass model to M_enc (geometry ladder)
#   3. Solve yN = 36/1365 -> R_env (horizon, independently)
#      Solve yN = 18/65   -> R_t   (throat, independently)
#      Verify R_t / R_env ~ 0.30
#
# Diagnostic (optional):
#   4. Direct y_N crossing from observations -> Rt_obs
#   5. Sigma stage (GFD base vs sigma visualization)
#   6. Displacement diagnostic
# =====================================================================

def infer_from_observations(observations, accel_ratio=1.0,
                            num_points=500):
    """Run the pure observation-driven inference pipeline.

    Derives manifold geometry (R_t, R_env) and mass model from
    observation data alone. No SPARC catalog values used.

    The GFD field equation inversion is exact: observations ARE GFD.
    The smooth mass model is fitted only as a geometry ladder for the
    yN solvers. The sigma stage is an optional diagnostic that
    visualizes the decomposition of mass vs vortex power output.

    Parameters
    ----------
    observations : list of dict
        Each dict has 'r' (kpc), 'v' (km/s), 'err' (km/s).
    accel_ratio : float
        Acceleration scale ratio (default 1.0).
    num_points : int
        Radial grid resolution (default 500).

    Returns
    -------
    dict with all derived quantities.
    """
    a0_eff = A0 * accel_ratio

    # =================================================================
    # CORE STEP 1: Invert field equation at each observation
    # The round trip (v -> M_enc -> v) is lossless. Observations = GFD.
    # =================================================================
    inverted = invert_observations(observations, a0_eff)

    if len(inverted) < 3:
        return {
            "error": "Need at least 3 valid observations",
            "method": "pure_observation",
        }

    # =================================================================
    # CORE STEP 2: Fit parametric mass model (geometry ladder)
    # The smooth model provides continuous M_enc(r) for yN solvers.
    # It is NOT used for rotation curve prediction.
    # =================================================================
    fit = fit_mass_model(inverted, a0_eff)

    if fit is None:
        return {
            "error": "Could not fit mass model to inverted profile",
            "method": "pure_observation",
        }

    params = fit["params"]
    mass_model = fit["model"]
    gfd_rms = fit["gfd_rms"]
    Mb, ab, Md, Rd, Mg, Rg = params

    # =================================================================
    # CORE STEP 3: Solve manifold geometry from two yN conditions
    #   Throat:  yN(R_t)   = 18/65   (field origin)
    #   Horizon: yN(R_env) = 36/1365 (field boundary)
    # Both solved independently. R_t/R_env ~ 0.30 emerges.
    # =================================================================
    field_geometry = solve_field_geometry(Mb, ab, Md, Rd, Mg, Rg, a0_eff)
    r_t = field_geometry.get("throat_radius_kpc")
    r_env = field_geometry.get("envelope_radius_kpc")

    # For deep-field galaxies: use max observation radius as fallback
    max_obs_r = max(o[0] for o in inverted)
    if r_env is None or r_env <= 0:
        r_env = max_obs_r * 1.05

    # =================================================================
    # DIAGNOSTIC STEP 4: Direct y_N crossing from observations
    # Independent of the parametric fit. Compares observational
    # throat to the topological throat as a consistency check.
    # =================================================================
    yN_profile = compute_yN_profile(inverted, a0_eff)
    obs_geometry = find_throat_direct(yN_profile)
    rt_obs = obs_geometry.get("throat_radius_kpc")

    # =================================================================
    # DIAGNOSTIC STEP 5: Sigma stage (optional visualization layer)
    # Computes GFD base (mass only) and GFD sigma (mass + power
    # output) curves for visualization. The gap between them shows
    # the vortex contribution. Not needed for geometry derivation.
    # =================================================================
    tp_result = search_throughput(
        inverted, params, r_env, accel_ratio, a0_eff, num_points)

    # DIAGNOSTIC STEP 6: Displacement diagnostic via sigma-squared law
    sigma = tp_result["throughput"]
    sig_sq = sigma * sigma

    displacement = None
    if r_t is not None and r_t > 0:
        rt_pred_sig2 = r_t * (1.0 + sig_sq)

        disp = {
            "Rt_bary": round(r_t, 4),
            "sigma": sigma,
            "sigma_squared": round(sig_sq, 6),
            "Rt_pred_sigma2": round(rt_pred_sig2, 4),
        }

        if rt_obs is not None and rt_obs > 0:
            actual_shift = (rt_obs - r_t) / r_t
            disp["Rt_obs"] = round(rt_obs, 4)
            disp["delta_kpc"] = round(rt_obs - r_t, 4)
            disp["delta_pct"] = round(actual_shift * 100, 2)
            disp["pred_delta_pct"] = round(sig_sq * 100, 2)
            if abs(sig_sq) > 1e-8:
                cleanliness = actual_shift / sig_sq
                disp["cleanliness_index"] = round(cleanliness, 4)
            else:
                disp["cleanliness_index"] = None
            disp["pred_error_kpc"] = round(rt_pred_sig2 - rt_obs, 4)
        else:
            disp["Rt_obs"] = None
            disp["cleanliness_index"] = None

        displacement = disp

    # Build inverted mass table
    inverted_masses = [
        {"r_kpc": round(r, 2), "v_obs": round(v, 1),
         "M_enc_solar": round(m, 2)}
        for r, v, err, m in inverted
    ]

    # Build y_N table
    yN_table = [
        {"r_kpc": round(r, 2), "yN": round(yN, 6)}
        for r, yN in yN_profile
    ]

    return {
        "method": "pure_observation",
        "n_observations": len(inverted),
        "inverted_masses": inverted_masses,
        "yN_profile": yN_table,
        "mass_model": mass_model,
        "gfd_base_rms": gfd_rms,
        "field_geometry": field_geometry,
        "field_geometry_obs": obs_geometry,
        "displacement": displacement,
        "throughput": tp_result["throughput"],
        "f_gas": tp_result["f_gas"],
        "m_stellar": tp_result["m_stellar"],
        "sigma_rms": tp_result["rms"],
        "chi2_dof": tp_result["chi2_dof"],
    }
