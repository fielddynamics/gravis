"""
Flask API routes for GRAVIS rotation curve analysis.

Endpoints:
  GET  /api/galaxies             - list all galaxies by mode
  GET  /api/galaxies/<id>        - get single galaxy details
  POST /api/rotation-curve       - compute rotation curves for given parameters
  POST /api/infer-mass           - infer mass from observed velocity
  POST /api/infer-mass-model     - infer scaled mass model from observation + shape
  POST /api/infer-mass-multi     - multi-point inference consistency analysis
"""

import math

from flask import Blueprint, jsonify, request

from physics import constants
from physics.mass_model import enclosed_mass, total_mass
from physics.newtonian import velocity as newtonian_velocity
from physics.aqual import velocity as dtg_velocity
from physics.mond import velocity as mond_velocity
from physics.inference import infer_mass
from physics.nfw import cdm_velocity, abundance_matching, fit_halo, concentration, r200_kpc
from data.galaxies import get_all_galaxies, get_galaxy_by_id

api = Blueprint("api", __name__, url_prefix="/api")


@api.route("/galaxies", methods=["GET"])
def list_galaxies():
    """Return all galaxies grouped by mode (prediction / inference)."""
    return jsonify(get_all_galaxies())


@api.route("/galaxies/<galaxy_id>", methods=["GET"])
def get_galaxy(galaxy_id):
    """Return a single galaxy by id."""
    galaxy = get_galaxy_by_id(galaxy_id)
    if galaxy is None:
        return jsonify({"error": "Galaxy not found"}), 404
    return jsonify(galaxy)


@api.route("/rotation-curve", methods=["POST"])
def compute_rotation_curve():
    """
    Compute Newtonian, DTG, MOND, and CDM rotation curves.

    Request JSON:
    {
        "max_radius": 30,          // kpc
        "num_points": 100,         // number of radial points
        "accel_ratio": 1.0,        // a/a0 multiplier
        "mass_model": {            // distributed mass model
            "bulge": {"M": 1.5e10, "a": 0.6},
            "disk":  {"M": 5.0e10, "Rd": 2.5},
            "gas":   {"M": 1.0e10, "Rd": 5.0}
        },
        "observations": [...]      // optional: for CDM halo fitting
    }

    Response JSON:
    {
        "radii": [...],            // kpc
        "newtonian": [...],        // km/s
        "dtg": [...],              // km/s
        "mond": [...],             // km/s
        "cdm": [...],              // km/s (baryonic + best-fit NFW halo)
        "enclosed_mass": [...],    // M_sun at each radius
        "cdm_halo": {...}          // NFW fit details (M200, c, chi2, etc.)
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    max_radius = data.get("max_radius", 30)
    num_points = data.get("num_points", 100)
    accel_ratio = data.get("accel_ratio", 1.0)
    mass_model = data.get("mass_model")
    observations = data.get("observations")

    if not mass_model:
        return jsonify({"error": "mass_model is required"}), 400

    # Validate num_points (cap at 500 for safety)
    num_points = min(int(num_points), 500)
    num_points = max(num_points, 10)

    # ------------------------------------------------------------------
    # Determine NFW halo mass: fit to observations or abundance matching
    # ------------------------------------------------------------------
    m_total = total_mass(mass_model)

    def mass_at_r(r):
        return enclosed_mass(r, mass_model)

    cdm_halo_info = None
    m200 = None

    if observations and len(observations) >= 2:
        # Best-fit NFW halo to observed rotation data
        fit = fit_halo(observations, mass_at_r, accel_ratio)
        if fit:
            m200 = fit['m200']
            cdm_halo_info = fit
            cdm_halo_info['method'] = 'chi-squared fit to observations'
    
    if m200 is None:
        # Fallback: abundance matching (Moster+ 2013)
        m200 = abundance_matching(m_total)
        c_val = concentration(m200)
        r200 = r200_kpc(m200)
        cdm_halo_info = {
            'm200': m200,
            'c': round(c_val, 2),
            'r200_kpc': round(r200, 2),
            'n_params_fitted': 0,
            'n_params_total': 2,
            'method': 'abundance matching (Moster+ 2013)',
        }

    # ------------------------------------------------------------------
    # Compute curves at each radius
    # ------------------------------------------------------------------
    radii = []
    newtonian = []
    dtg = []
    mond = []
    cdm = []
    enc_mass = []

    for i in range(num_points):
        r = (max_radius / num_points) * (i + 1)
        m_at_r = enclosed_mass(r, mass_model)

        radii.append(round(r, 6))
        enc_mass.append(round(m_at_r, 2))
        newtonian.append(round(newtonian_velocity(r, m_at_r), 4))
        dtg.append(round(dtg_velocity(r, m_at_r, accel_ratio), 4))
        mond.append(round(mond_velocity(r, m_at_r, accel_ratio), 4))
        cdm.append(round(cdm_velocity(r, m_at_r, m200), 4))

    # Format halo info for JSON
    if cdm_halo_info:
        cdm_halo_info['m200'] = round(cdm_halo_info['m200'], 2)

    return jsonify({
        "radii": radii,
        "newtonian": newtonian,
        "dtg": dtg,
        "mond": mond,
        "cdm": cdm,
        "enclosed_mass": enc_mass,
        "cdm_halo": cdm_halo_info,
    })


@api.route("/infer-mass", methods=["POST"])
def infer_mass_endpoint():
    """
    Infer enclosed baryonic mass from observed velocity.

    Request JSON:
    {
        "r_kpc": 8.0,             // galactocentric radius
        "v_km_s": 230.0,          // observed circular velocity
        "accel_ratio": 1.0        // a/a0 multiplier
    }

    Response JSON:
    {
        "inferred_mass_solar": 1.23e11,
        "log10_mass": 11.09
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    r_kpc = data.get("r_kpc", 8.0)
    v_km_s = data.get("v_km_s", 230.0)
    accel_ratio = data.get("accel_ratio", 1.0)

    mass = infer_mass(r_kpc, v_km_s, accel_ratio)

    return jsonify({
        "inferred_mass_solar": mass,
        "log10_mass": round(math.log10(max(mass, 1.0)), 4),
    })


@api.route("/infer-mass-model", methods=["POST"])
def infer_mass_model_endpoint():
    """
    Infer a scaled mass model from an observation and a distribution shape.

    Given an observed (r, v) and a mass model "shape" (scale lengths +
    relative proportions), compute the total mass that makes DTG predict
    exactly v at r, then scale all components proportionally.

    Also returns the BTFR (Baryonic Tully-Fisher) mass estimate:
      M_BTFR = v^4 / (G * a0)

    Request JSON:
    {
        "r_kpc": 8.0,
        "v_km_s": 230.0,
        "accel_ratio": 1.0,
        "mass_model": {
            "bulge": {"M": 1.5e10, "a": 0.6},
            "disk":  {"M": 5.0e10, "Rd": 2.5},
            "gas":   {"M": 1.0e10, "Rd": 5.0}
        }
    }

    Response JSON:
    {
        "inferred_enclosed": 5.5e10,
        "scale_factor": 0.95,
        "inferred_mass_model": {
            "bulge": {"M": 1.425e10, "a": 0.6},
            "disk":  {"M": 4.75e10, "Rd": 2.5},
            "gas":   {"M": 0.95e10, "Rd": 5.0}
        },
        "inferred_total": 7.125e10,
        "log10_total": 10.853,
        "btfr_mass": 7.0e10,
        "log10_btfr": 10.845
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    r_kpc = data.get("r_kpc", 8.0)
    v_km_s = data.get("v_km_s", 230.0)
    accel_ratio = data.get("accel_ratio", 1.0)
    mass_model = data.get("mass_model")

    if not mass_model:
        return jsonify({"error": "mass_model is required"}), 400

    # Step 1: Infer M(<r) from the observation using DTG inverse
    m_enclosed_needed = infer_mass(r_kpc, v_km_s, accel_ratio)

    # Step 2: Compute M(<r) for the current mass model shape
    m_enclosed_current = enclosed_mass(r_kpc, mass_model)
    m_total_current = total_mass(mass_model)

    if m_enclosed_current <= 0 or m_total_current <= 0:
        return jsonify({"error": "Mass model has zero enclosed mass at observation radius"}), 400

    # Step 3: Scale factor to match observation
    scale = m_enclosed_needed / m_enclosed_current

    # Step 4: Build scaled mass model (preserve scale lengths, scale masses)
    scaled_model = {}
    for comp in ("bulge", "disk", "gas"):
        if comp in mass_model and mass_model[comp].get("M", 0) > 0:
            entry = dict(mass_model[comp])
            entry["M"] = entry["M"] * scale
            scaled_model[comp] = entry
        elif comp in mass_model:
            scaled_model[comp] = dict(mass_model[comp])

    inferred_total = total_mass(scaled_model)

    # Step 5: BTFR mass estimate: M = v^4 / (G * a0)
    v_ms = v_km_s * 1000.0
    a0_eff = constants.A0 * accel_ratio
    btfr_mass = (v_ms ** 4) / (constants.G * a0_eff) / constants.M_SUN

    return jsonify({
        "inferred_enclosed": round(m_enclosed_needed, 2),
        "scale_factor": round(scale, 6),
        "inferred_mass_model": scaled_model,
        "inferred_total": round(inferred_total, 2),
        "log10_total": round(math.log10(max(inferred_total, 1.0)), 4),
        "btfr_mass": round(btfr_mass, 2),
        "log10_btfr": round(math.log10(max(btfr_mass, 1.0)), 4),
    })


@api.route("/infer-mass-multi", methods=["POST"])
def infer_mass_multi_endpoint():
    """
    Infer total baryonic mass from multiple (r, v) observation points.

    For each point, infers the total mass using the mass model shape,
    then reports per-point results and aggregate statistics.

    Request JSON:
    {
        "observations": [{"r": 5, "v": 236}, {"r": 8, "v": 230}, ...],
        "accel_ratio": 1.0,
        "mass_model": { ... }
    }

    Response includes per-point inferred totals plus mean, std, and
    reduced chi-squared of the consistency across radii.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    observations = data.get("observations", [])
    accel_ratio = data.get("accel_ratio", 1.0)
    mass_model = data.get("mass_model")

    if not mass_model or not observations:
        return jsonify({"error": "mass_model and observations are required"}), 400

    results = []
    totals = []
    weights = []

    m_total_current = total_mass(mass_model)
    if m_total_current <= 0:
        return jsonify({"error": "Mass model has zero total mass"}), 400

    mass_errors = []  # per-point mass uncertainty from velocity error bars

    for obs in observations:
        r_kpc = obs.get("r", 0)
        v_km_s = obs.get("v", 0)
        err = obs.get("err", 0)
        if r_kpc <= 0 or v_km_s <= 0:
            continue

        # Infer enclosed mass from this (r, v) point
        m_enclosed_needed = infer_mass(r_kpc, v_km_s, accel_ratio)

        # Compute scale factor relative to the mass model shape
        m_enclosed_current = enclosed_mass(r_kpc, mass_model)

        if m_enclosed_current <= 0:
            continue

        scale = m_enclosed_needed / m_enclosed_current
        inferred_total = m_total_current * scale

        # Enclosed fraction: how much of the model mass is within this radius
        enc_frac = m_enclosed_current / m_total_current

        # Propagated error: infer mass at v +/- err to get delta_M
        sigma_err = max(err, 1.0)
        m_hi = infer_mass(r_kpc, v_km_s + sigma_err, accel_ratio)
        m_lo = infer_mass(r_kpc, max(v_km_s - sigma_err, 1.0), accel_ratio)
        scale_hi = m_hi / m_enclosed_current
        scale_lo = m_lo / m_enclosed_current
        total_hi = m_total_current * scale_hi
        total_lo = m_total_current * scale_lo
        delta_m = abs(total_hi - total_lo) / 2.0

        # Compute GFD predicted velocity at this radius using the MODEL mass
        # (not the inferred mass -- we compare observation to the prediction)
        v_gfd = dtg_velocity(r_kpc, m_enclosed_current, accel_ratio)
        delta_v = round(v_km_s - v_gfd, 2)
        sigma_dev = round(abs(delta_v) / max(err, 0.1), 2) if err > 0 else None

        results.append({
            "r_kpc": r_kpc,
            "v_km_s": v_km_s,
            "err": err,
            "inferred_total": round(inferred_total, 2),
            "log10_total": round(math.log10(max(inferred_total, 1.0)), 4),
            "enclosed_frac": round(enc_frac, 4),
            "v_gfd": round(v_gfd, 2),
            "delta_v": delta_v,
            "sigma_dev": sigma_dev,
        })
        totals.append(inferred_total)
        weights.append(enc_frac)
        mass_errors.append(delta_m)

    if len(totals) < 2:
        return jsonify({"error": "Need at least 2 valid observation points"}), 400

    n = len(totals)

    # Unweighted statistics
    mean_total = sum(totals) / n
    variance = sum((t - mean_total) ** 2 for t in totals) / (n - 1)
    std_total = math.sqrt(variance)
    cv = (std_total / mean_total) * 100.0 if mean_total > 0 else 0.0

    # Weighted statistics (weight = enclosed mass fraction)
    w_sum = sum(weights)
    if w_sum > 0:
        w_mean = sum(t * w for t, w in zip(totals, weights)) / w_sum
        w_sum2 = sum(w * w for w in weights)
        if w_sum * w_sum - w_sum2 > 0:
            w_var = (w_sum / (w_sum * w_sum - w_sum2)) * \
                    sum(w * (t - w_mean) ** 2 for t, w in zip(totals, weights))
            w_std = math.sqrt(max(w_var, 0))
        else:
            w_std = 0.0
        w_cv = (w_std / w_mean) * 100.0 if w_mean > 0 else 0.0
    else:
        w_mean = mean_total
        w_std = std_total
        w_cv = cv

    # -----------------------------------------------------------------
    # Band methods: 5 different half-width calculations (in M_sun)
    # Each represents a different statistical view of the uncertainty.
    # The frontend applies these as +/- around the anchor mass.
    # Note: the band is always centered on the GFD anchor (modelTotal),
    # so half-widths must be computed relative to the anchor, not the
    # data midpoint.
    # -----------------------------------------------------------------
    sorted_totals = sorted(totals)

    # 1. Weighted RMS from anchor: captures scatter + systematic offset
    #    Weight = enclosed fraction. The current default.
    #    Not returned here -- computed on frontend since it needs modelTotal.

    # 2. 1-sigma Weighted Scatter: pure spread of per-point estimates
    band_scatter = w_std

    # 3. Propagated Observational Error: formal error from velocity bars
    #    Combine per-point mass errors weighted by enclosed fraction.
    #    This computes the weighted RMS of per-point mass uncertainties,
    #    giving the "typical" mass error from measurement noise alone.
    if w_sum > 0 and mass_errors:
        w_err_sq = sum(w * de * de for w, de in zip(weights, mass_errors))
        band_obs_err = math.sqrt(w_err_sq / w_sum)
    else:
        band_obs_err = std_total

    # 4. Min-Max Envelope: computed on frontend since it needs modelTotal
    #    (the GFD anchor mass). Backend returns raw min/max for the
    #    frontend to compute max(|M_i - anchor|).

    # 5. IQR (Robust): interquartile range via linear interpolation.
    #    Resistant to outlier inner points with low enclosed fraction.
    def _percentile(data, p):
        """Linear interpolation percentile (same as numpy default)."""
        k = (len(data) - 1) * p
        f = int(k)
        c = min(f + 1, len(data) - 1)
        d = k - f
        return data[f] + d * (data[c] - data[f])

    q1 = _percentile(sorted_totals, 0.25)
    q3 = _percentile(sorted_totals, 0.75)
    band_iqr = (q3 - q1) / 2.0

    # -----------------------------------------------------------------
    # Shape diagnostic: inner vs outer half velocity residual summary
    # Only include points with enclosed_frac >= 0.05 for reliability
    # -----------------------------------------------------------------
    diag_pts = [p for p in results
                if p.get("enclosed_frac", 0) >= 0.05
                and p.get("delta_v") is not None
                and p.get("sigma_dev") is not None]
    shape_diagnostic = None
    if len(diag_pts) >= 4:
        mid = len(diag_pts) // 2
        inner = diag_pts[:mid]
        outer = diag_pts[mid:]
        shape_diagnostic = {
            "inner_r_max": inner[-1]["r_kpc"],
            "outer_r_min": outer[0]["r_kpc"],
            "inner_mean_dv": round(
                sum(p["delta_v"] for p in inner) / len(inner), 2),
            "outer_mean_dv": round(
                sum(p["delta_v"] for p in outer) / len(outer), 2),
            "inner_mean_sigma": round(
                sum(p["sigma_dev"] for p in inner) / len(inner), 2),
            "outer_mean_sigma": round(
                sum(p["sigma_dev"] for p in outer) / len(outer), 2),
            "n_inner": len(inner),
            "n_outer": len(outer),
        }

    return jsonify({
        "points": results,
        "n_points": n,
        "mean_total": round(mean_total, 2),
        "std_total": round(std_total, 2),
        "log10_mean": round(math.log10(max(mean_total, 1.0)), 4),
        "cv_percent": round(cv, 2),
        "weighted_mean": round(w_mean, 2),
        "weighted_std": round(w_std, 2),
        "log10_weighted_mean": round(math.log10(max(w_mean, 1.0)), 4),
        "weighted_cv_percent": round(w_cv, 2),
        "min_total": round(min(totals), 2),
        "max_total": round(max(totals), 2),
        "band_methods": {
            "weighted_scatter": round(band_scatter, 2),
            "obs_error": round(band_obs_err, 2),
            "iqr": round(band_iqr, 2),
        },
        "shape_diagnostic": shape_diagnostic,
    })


@api.route("/constants", methods=["GET"])
def get_constants():
    """Return the physical constants used by the engine."""
    return jsonify({
        "G": constants.G,
        "M_SUN": constants.M_SUN,
        "KPC_TO_M": constants.KPC_TO_M,
        "M_E": constants.M_E,
        "R_E": constants.R_E,
        "K_SIMPLEX": constants.K_SIMPLEX,
        "A0": constants.A0,
    })
