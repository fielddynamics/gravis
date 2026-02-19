"""
Vortex Service: velocity curves and transformation from observations.

Two methods, one per image:
  Method 1 (figure_a): Raw velocity curve + optional fractional-variance curve
    over observation span. If variance_pct is 0, skip FV (chart 2 = chart 1).
  Method 2 (figure_b): Transformation T(r) = M_derived/M_phot. Chart 1 from raw
    velocity; chart 2 from the velocity produced by method 1 (so method 2
    uses method 1, no duplicated logic).

Variance: variance_pct (default 1.5). If 0, fractional variance step is skipped.
  If > 0, s = (variance_pct/100) * sum((v - mean)^2) for the smoothing spline.

Endpoints:
  POST /api/vortex/figure-a  -> { figure_a: { chart1, chart2 }, ... }
  POST /api/vortex/figure-b  -> { figure_b: { chart1, chart2 } } (calls method 1)

No unicode (Windows charmap).
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from flask import jsonify, request

from physics.services import GravisService
from physics.constants import A0, KPC_TO_M, G, M_SUN
from physics.mass_model import enclosed_mass
from data.galaxies import get_galaxy_by_id

DEFAULT_VARIANCE_PCT = 1.5
G_SI = G
MSUN_KG = M_SUN


def _value_at_radius(radii, v_curve, r):
    """Linear interpolation: v at r from (radii, v_curve). Returns float or None."""
    radii = np.asarray(radii, dtype=float)
    v_curve = np.asarray(v_curve, dtype=float)
    if len(radii) != len(v_curve) or len(radii) == 0:
        return None
    if r <= radii[0]:
        return float(v_curve[0])
    if r >= radii[-1]:
        return float(v_curve[-1])
    return float(np.interp(r, radii, v_curve))


def get_gfd_velocity_bridge(radii, v_raw):
    """
    GFD raw velocity at r = -1, 0, 1 kpc (covariant-completion anchor for the bridge).

    Used to pin all theory curves (MOND, Newtonian, CDM, GFD Photometric) to the
    same core value so the bridge through the throat is well-defined.

    Parameters
    ----------
    radii : array-like
        Radii (kpc), can include negative (symmetric vortex grid).
    v_raw : array-like
        GFD raw velocity at each radius (same length as radii).

    Returns
    -------
    tuple (v_at_minus_1, v_at_0, v_at_1) or None if interpolation fails.
    """
    if not radii or not v_raw or len(radii) != len(v_raw):
        return None
    radii = np.asarray(radii, dtype=float)
    v_raw = np.asarray(v_raw, dtype=float)
    r_abs_max = float(np.max(np.abs(radii)))
    if r_abs_max < 1.0:
        return None
    v_m1 = _value_at_radius(radii, v_raw, -1.0)
    v0 = _value_at_radius(radii, v_raw, 0.0)
    v1 = _value_at_radius(radii, v_raw, 1.0)
    if v_m1 is None or v0 is None or v1 is None:
        return None
    return (float(v_m1), float(v0), float(v1))


def get_gfd_velocity_bridge_from_observations(observations, max_radius):
    """
    Build GFD raw curve from observations and return bridge anchor at r = -1, 0, 1.

    Parameters
    ----------
    observations : list of dict with "r", "v"
    max_radius : float (kpc)

    Returns
    -------
    tuple (v_at_minus_1, v_at_0, v_at_1) or None.
    """
    if not observations or len(observations) < 3:
        return None
    obs_r = np.array([float(o["r"]) for o in observations])
    obs_v = np.array([float(o["v"]) for o in observations])
    num_points = 500
    radii_fine = np.linspace(-float(max_radius), float(max_radius), num_points)
    v_raw = _smooth_velocity_curve(obs_r, obs_v, radii_fine, smoothing=0.0)
    return get_gfd_velocity_bridge(radii_fine.tolist(), v_raw.tolist())


def _smooth_velocity_curve(obs_r, obs_v, radii_fine, smoothing=0.0):
    """Build v(r) from observation points. For negative r use v(-r)=v(r)."""
    obs_r = np.asarray(obs_r, dtype=float)
    obs_v = np.asarray(obs_v, dtype=float)
    order = np.argsort(obs_r)
    r_sorted = obs_r[order]
    v_sorted = obs_v[order]
    r_min, r_max = float(r_sorted[0]), float(r_sorted[-1])
    spl = UnivariateSpline(r_sorted, v_sorted, s=smoothing, k=min(3, len(obs_r) - 1))
    v_fine = np.zeros_like(radii_fine)
    for i, r in enumerate(radii_fine):
        r_abs = abs(r)
        if r_abs <= r_min:
            v_fine[i] = v_sorted[0]
        elif r_abs >= r_max:
            v_fine[i] = v_sorted[-1]
        else:
            v_fine[i] = float(spl(r_abs))
    return v_fine


def _compute_transformation(radii_pos, v_pos, mass_model, a0):
    """T(r) = M_derived/M_phot on positive r. Returns dict or None."""
    if not mass_model or a0 <= 0:
        return None
    r_kpc = np.asarray(radii_pos, dtype=float)
    v_km_s = np.asarray(v_pos, dtype=float)
    if len(r_kpc) == 0:
        return None
    r_m = r_kpc * KPC_TO_M
    v_ms = v_km_s * 1000.0
    g_obs = (v_ms * v_ms) / np.maximum(r_m, 1e-30)
    x = (g_obs / a0) ** 2
    f_x = 1.0 + x + x ** 2
    g_baryon = g_obs / np.maximum(f_x, 1e-30)
    M_derived_kg = g_baryon * (r_m ** 2) / G_SI
    M_derived_solar = M_derived_kg / MSUN_KG
    M_phot_solar = np.array([enclosed_mass(float(r), mass_model) for r in r_kpc], dtype=float)
    ratio_T = np.where(M_phot_solar > 0, M_derived_solar / M_phot_solar, np.nan)
    return {"r_kpc": r_kpc, "ratio_T": ratio_T}


def method_1_figure_a(galaxy_id, variance_pct=0.0):
    """
    Method 1: Figure A data (velocity curves).
    variance_pct: 0 = skip FV (chart 2 same as chart 1); >0 = apply FV (e.g. 1.5).
    Returns dict with figure_a, radii_fine, v_raw, v_fv_on_raw, obs, mass_model, etc.
    """
    gal = get_galaxy_by_id(galaxy_id)
    if not gal:
        return None
    observations = gal.get("observations", [])
    if len(observations) < 3:
        return None

    obs_r = np.array([float(o["r"]) for o in observations])
    obs_v = np.array([float(o["v"]) for o in observations])
    obs_err = [max(float(o.get("err", 5.0)), 1.0) for o in observations]

    r_min_obs = float(np.min(obs_r))
    r_max_obs = float(np.max(obs_r))
    chart_max = r_max_obs * 1.15
    num_points = 500
    radii_fine = np.linspace(-chart_max, chart_max, num_points)

    v_raw = _smooth_velocity_curve(obs_r, obs_v, radii_fine, smoothing=0.0)

    if variance_pct <= 0:
        v_fv_on_raw = v_raw
    else:
        frac = variance_pct / 100.0
        v_mean_raw = np.mean(v_raw)
        s_param = float(frac * np.sum((v_raw - v_mean_raw) ** 2))
        spl = UnivariateSpline(radii_fine, v_raw, s=s_param, k=min(3, num_points - 1))
        v_fv_on_raw = spl(radii_fine)

    in_obs_span = (radii_fine >= -r_max_obs) & (radii_fine <= r_max_obs)
    v_fv_display = np.where(in_obs_span, v_fv_on_raw, np.nan)

    figure_a = {
        "chart1": {
            "radii": radii_fine.tolist(),
            "v_raw": v_raw.tolist(),
            "obs_r": obs_r.tolist(),
            "obs_v": obs_v.tolist(),
            "obs_err": obs_err,
        },
        "chart2": {
            "radii": radii_fine.tolist(),
            "v_raw": v_raw.tolist(),
            "v_fv": [float(x) if np.isfinite(x) else None for x in v_fv_display],
            "obs_r": obs_r.tolist(),
            "obs_v": obs_v.tolist(),
            "obs_err": obs_err,
            "variance_pct": variance_pct,
        },
    }

    mask_pos = radii_fine > 0
    radii_pos = radii_fine[mask_pos]
    mass_model = gal.get("mass_model")

    return {
        "figure_a": figure_a,
        "galaxy_id": galaxy_id,
        "radii_fine": radii_fine.tolist(),
        "v_raw": v_raw.tolist(),
        "v_fv_on_raw": v_fv_on_raw.tolist(),
        "radii_pos": radii_pos.tolist(),
        "mass_model": mass_model,
        "obs_r": obs_r.tolist(),
        "obs_v": obs_v.tolist(),
        "obs_err": obs_err,
    }


def method_2_figure_b(galaxy_id, variance_pct=0.0):
    """
    Method 2: Figure B data (transformation). Uses method 1 output.
    """
    out = method_1_figure_a(galaxy_id, variance_pct)
    if not out:
        return None

    radii_fine = np.array(out["radii_fine"])
    v_raw = np.array(out["v_raw"])
    v_fv_on_raw = np.array(out["v_fv_on_raw"])
    radii_pos = np.array(out["radii_pos"])
    mass_model = out["mass_model"]

    mask_pos = radii_fine > 0
    trans_raw = _compute_transformation(radii_pos, v_raw[mask_pos], mass_model, A0)
    trans_fv = _compute_transformation(radii_pos, v_fv_on_raw[mask_pos], mass_model, A0)

    figure_b = {
        "chart1": {
            "r_kpc": trans_raw["r_kpc"].tolist() if trans_raw else [],
            "ratio_T": trans_raw["ratio_T"].tolist() if trans_raw else [],
        },
        "chart2": {
            "r_kpc": trans_fv["r_kpc"].tolist() if trans_fv else [],
            "ratio_T": trans_fv["ratio_T"].tolist() if trans_fv else [],
            "variance_pct": variance_pct,
        },
    }

    return {
        "figure_b": figure_b,
        "galaxy_id": galaxy_id,
    }


class VortexService(GravisService):
    """Vortex velocity curves and transformation. Two methods; method 2 uses method 1."""

    id = "vortex"
    name = "Vortex"
    description = "Velocity curves and transformation (Figure A/B) with optional fractional variance"
    category = "galactic"
    status = "live"
    route = "/analysis"

    def validate(self, config):
        if not config:
            raise ValueError("Request body must be JSON")
        galaxy_id = config.get("galaxy_id")
        if not galaxy_id:
            raise ValueError("galaxy_id is required")
        variance_pct = config.get("variance_pct", DEFAULT_VARIANCE_PCT)
        variance_pct = float(variance_pct) if variance_pct is not None else DEFAULT_VARIANCE_PCT
        if variance_pct < 0:
            variance_pct = 0.0
        return {"galaxy_id": galaxy_id.strip().lower(), "variance_pct": variance_pct}

    def compute(self, config):
        """Not used; we expose figure-a and figure-b via register_routes."""
        raise NotImplementedError("Use POST /api/vortex/figure-a or /api/vortex/figure-b")

    def register_routes(self, bp):
        service = self

        @bp.route("/vortex/figure-a", methods=["POST"])
        def vortex_figure_a():
            data = request.get_json()
            try:
                config = service.validate(data or {})
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            out = method_1_figure_a(config["galaxy_id"], config["variance_pct"])
            if not out:
                return jsonify({"error": "Galaxy not found or too few observations"}), 404
            return jsonify({
                "figure_a": out["figure_a"],
                "galaxy_id": out["galaxy_id"],
            })

        @bp.route("/vortex/figure-b", methods=["POST"])
        def vortex_figure_b():
            data = request.get_json()
            try:
                config = service.validate(data or {})
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            out = method_2_figure_b(config["galaxy_id"], config["variance_pct"])
            if not out:
                return jsonify({"error": "Galaxy not found or too few observations"}), 404
            return jsonify({
                "figure_b": out["figure_b"],
                "galaxy_id": out["galaxy_id"],
            })
