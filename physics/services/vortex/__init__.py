"""
Vortex Service: velocity curves and transformation from observations.

Uses the same GFD field velocity as sst_topological_velocity / gfd_field_velocity_nails_it:
  Observation -> defraction (6.2% Gaussian kernel) -> field velocity (covariant + Eq 75 Poisson).
  No R_t, R_env; direct field reading.

Two methods, one per image:
  Method 1 (figure_a): Field velocity from defraction + optional fractional-variance curve.
  Method 2 (figure_b): Transformation T(r) = M_derived/M_phot using method 1 curves.

Variance: variance_pct (default 1.5). If > 0, apply FV smoothing on top of defraction curve.

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
from physics.sst_topological_velocity import gfd_velocity_curve_sst, gfd_sst_velocity_decode_curve
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
    Purple "GFD (Velocity smooth)" = forward from mass profile only (Eq 75 + SST). Same as panel script.
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

    r_max_obs = float(np.max(obs_r))
    chart_max = r_max_obs * 1.15
    num_points = 500
    radii_fine = np.linspace(-chart_max, chart_max, num_points)

    # Grey dotted "GFD (Velocity raw)": unsmoothed curve from observations (spline interpolation only)
    v_raw = _smooth_velocity_curve(obs_r, obs_v, radii_fine, smoothing=0.0)

    # Purple "GFD (Velocity smooth)": GFD from mass profile (Eq 75 + SST). Start at 0.5 kpc so no ramp from zero (match Python script).
    GFD_MIN_R_KPC = 0.5
    mask_pos = radii_fine > 0
    radii_pos = radii_fine[mask_pos]
    r_abs = np.abs(radii_fine)
    mass_model = gal.get("mass_model")
    if mass_model and len(radii_pos) > 0:
        radii_gfd = radii_pos[radii_pos >= GFD_MIN_R_KPC]
        if len(radii_gfd) > 0:
            v_field_pos = np.array(gfd_velocity_curve_sst(
                radii_gfd.tolist(), lambda r, mm=mass_model: enclosed_mass(r, mm), A0), dtype=float)
            v_at_min = float(v_field_pos[0])
            v_fv_on_raw = np.where(r_abs < GFD_MIN_R_KPC, v_at_min, np.interp(r_abs, radii_gfd, v_field_pos))
        else:
            v_field_pos = np.array([], dtype=float)
            v_fv_on_raw = np.full_like(radii_fine, np.nan)
    else:
        v_field_pos = np.array([], dtype=float)
        v_fv_on_raw = np.full_like(radii_fine, np.nan)

    # GFD (SST+Poisson photometric): full curve from mass model (disk+bulge+gas). No bridge: tiny gap at origin (~2%).
    v_fv_display = np.where(np.isfinite(v_fv_on_raw), v_fv_on_raw, np.nan)
    GFD_FV_GAP_R_KPC = 0.02
    gap_mask_fv = np.abs(radii_fine) < GFD_FV_GAP_R_KPC
    v_fv_display = np.where(gap_mask_fv, np.nan, v_fv_display)

    # GFD velocity decode (no mass input): obs -> defraction -> field velocity. Same as observational chart.
    # Clip to observation range: only show decode between first and last observation radius (and mirror).
    # Do not bridge mirror (r < 0) and real (r > 0): tiny gap at origin (~2%).
    GFD_DECODE_GAP_R_KPC = 0.02
    r_min_obs = float(np.min(obs_r))
    v_decode_display = np.full_like(radii_fine, np.nan)
    if len(obs_r) >= 2:
        radii_decode = radii_pos[radii_pos >= GFD_MIN_R_KPC]
        if len(radii_decode) > 0:
            v_decode_pos = gfd_sst_velocity_decode_curve(
                obs_r.tolist(), obs_v.tolist(), radii_decode.tolist(), A0)
            if v_decode_pos:
                v_decode_pos = np.array(v_decode_pos, dtype=float)
                v_at_min_d = float(v_decode_pos[0])
                v_decode_on_full = np.where(
                    r_abs < GFD_MIN_R_KPC, v_at_min_d,
                    np.interp(r_abs, radii_decode, v_decode_pos))
                v_decode_display = np.where(np.isfinite(v_decode_on_full), v_decode_on_full, np.nan)
                # Gap at origin: no bridge between mirror and real. Set NaN for |r| < GFD_DECODE_GAP_R_KPC.
                gap_mask = np.abs(radii_fine) < GFD_DECODE_GAP_R_KPC
                v_decode_display = np.where(gap_mask, np.nan, v_decode_display)
                # Clip to observation range: real side r_min_obs <= r <= r_max_obs; mirror side -r_max_obs <= r <= -r_min_obs.
                out_of_range = (
                    (radii_fine > 0) & ((radii_fine < r_min_obs) | (radii_fine > r_max_obs))
                    | (radii_fine < 0) & ((radii_fine < -r_max_obs) | (radii_fine > -r_min_obs))
                )
                v_decode_display = np.where(out_of_range, np.nan, v_decode_display)

    def _safe_vlist(arr):
        return [float(x) if np.isfinite(x) else None for x in np.asarray(arr)]

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
            "v_fv": _safe_vlist(v_fv_display),
            "gfd_sst_velocity_decode": _safe_vlist(v_decode_display),
            "obs_r": obs_r.tolist(),
            "obs_v": obs_v.tolist(),
            "obs_err": obs_err,
            "variance_pct": variance_pct,
        },
    }

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

    def _safe_tolist(arr):
        """Convert array to list, replacing nan with None for JSON serialization."""
        if arr is None or len(arr) == 0:
            return []
        return [None if not np.isfinite(x) else float(x) for x in np.asarray(arr)]

    figure_b = {
        "chart1": {
            "r_kpc": trans_raw["r_kpc"].tolist() if trans_raw else [],
            "ratio_T": _safe_tolist(trans_raw["ratio_T"]) if trans_raw else [],
        },
        "chart2": {
            "r_kpc": trans_fv["r_kpc"].tolist() if trans_fv else [],
            "ratio_T": _safe_tolist(trans_fv["ratio_T"]) if trans_fv else [],
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

    def figure_a(self, galaxy_id, variance_pct=0.0):
        """Return Figure A data (velocity curves). Same as method_1_figure_a. Used by RotationService charts."""
        return method_1_figure_a(galaxy_id, variance_pct)

    def figure_b(self, galaxy_id, variance_pct=0.0):
        """Return Figure B data (transformation). Same as method_2_figure_b. Used by RotationService charts."""
        return method_2_figure_b(galaxy_id, variance_pct)

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
