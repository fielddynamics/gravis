"""
Rotation Curves Service for GRAVIS.

Implements the GravisService interface for galactic rotation curve
analysis. This service wraps the existing GravisEngine pipeline and
owns all rotation-specific API endpoints under /api/rotation/*.

Endpoints:
    POST /api/rotation/photometric-curve - unified photometric chart (one engine run, all theory curves)
    POST /api/rotation/gfd-velocity-curve-v2 - velocity curve v2 (deflection = SST direct reading, no R_t/R_env)
    POST /api/rotation/curve           - compute rotation curves (prediction)
    POST /api/rotation/inference       - compute rotation curves (optimized)
    POST /api/rotation/infer-curve     - (legacy alias for /inference)
    POST /api/rotation/infer-mass      - single-point mass inference
    POST /api/rotation/infer-mass-model - infer scaled mass model
    POST /api/rotation/infer-mass-multi - multi-point inference
    GET  /api/rotation/galaxies        - list all galaxies (prediction + inference)
    GET  /api/rotation/galaxy-ids      - list all galaxy ids we have (for exclusion lists)
    GET  /api/rotation/galaxies/<id>   - get single galaxy by id

IMPORTANT: No unicode characters allowed (Windows charmap constraint).
"""

import math

import numpy as np
from scipy.interpolate import UnivariateSpline
from flask import jsonify, request

from physics.services import GravisService
from physics.engine import GravisConfig, GravisEngine, compute_fit_metrics
from physics.sigma import auto_vortex_strength
from physics.services.rotation.inference import optimize_inference, solve_field_geometry
from physics.services.rotation.field_analysis import compute_field_analysis
from physics import constants
from physics.mass_model import enclosed_mass, total_mass
from physics.inference import infer_mass
from physics.aqual import velocity as dtg_velocity
from physics.nfw import (
    abundance_matching, fit_halo, concentration, r200_kpc,
    m200_from_nfw_rho0_rs, LITERATURE_NFW_RHO0_RS,
)
from data.galaxies import get_all_galaxies, get_galaxy_by_id, get_galaxy_catalog
from physics.services.rotation.photometry import derive_mass_parameters_from_photometry
from physics.services.rotation.vortex_bridge import (
    mirror_curve_with_cutoff,
    truncate_to_first_obs,
)
from physics.sst_topological_velocity import (
    M_enc_from_velocity,
    g_source_from_velocity,
    gfd_velocity_curve_sst,
    gfd_sst_velocity_decode_curve,
)


def _interp(radii, values, r):
    """Linear interpolation: value at r from (radii, values). Returns None if out of range."""
    if not radii or not values or len(radii) != len(values):
        return None
    if r <= radii[0]:
        return float(values[0])
    if r >= radii[-1]:
        return float(values[-1])
    for i in range(len(radii) - 1):
        if radii[i] <= r <= radii[i + 1]:
            t = (r - radii[i]) / (radii[i + 1] - radii[i])
            return float(values[i] + t * (values[i + 1] - values[i]))
    return None


def _smooth_observed_velocity(obs_r, obs_v, target_radii, smoothing=0.0, variance_pct=0.0):
    """Smooth observed (r, v) onto target_radii. Same logic as vortex GFD (Velocity).

    If variance_pct > 0, applies fractional-variance smoothing (e.g. 6.2 for 6.2%):
    s_param = (variance_pct/100) * sum((v - mean)^2). Returns list of velocities
    at each target radius (extrapolate at ends).
    """
    if not obs_r or not obs_v or len(obs_r) != len(obs_v) or len(obs_r) < 2:
        return []
    obs_r = np.asarray(obs_r, dtype=float)
    obs_v = np.asarray(obs_v, dtype=float)
    order = np.argsort(obs_r)
    r_sorted = obs_r[order]
    v_sorted = obs_v[order]
    r_min, r_max = float(r_sorted[0]), float(r_sorted[-1])
    k = min(3, len(obs_r) - 1)
    spl = UnivariateSpline(r_sorted, v_sorted, s=smoothing, k=k)
    target = np.asarray(target_radii, dtype=float)
    v_raw = np.zeros_like(target)
    for i, r in enumerate(target):
        r = float(r)
        if r <= r_min:
            v_raw[i] = v_sorted[0]
        elif r >= r_max:
            v_raw[i] = v_sorted[-1]
        else:
            v_raw[i] = float(spl(r))

    if variance_pct <= 0:
        return [round(float(v), 4) for v in v_raw]

    frac = variance_pct / 100.0
    v_mean = float(np.mean(v_raw))
    s_param = float(frac * np.sum((v_raw - v_mean) ** 2))
    n_pts = len(target)
    k_fv = min(3, n_pts - 1) if n_pts >= 4 else min(2, n_pts - 1)
    spl_fv = UnivariateSpline(target, v_raw, s=s_param, k=k_fv)
    out = [round(float(spl_fv(r)), 4) for r in target]
    return out


class RotationService(GravisService):
    """
    Galactic rotation curve service.

    Provides zero-parameter rotation curve predictions using Dual Tetrad
    Gravity, plus Newtonian, MOND, and CDM comparisons. Also provides
    mass inference from observed velocities.
    """

    id = "rotation"
    name = "Rotation Curves"
    description = "Zero-parameter galactic rotation predictions using DTG"
    category = "galactic"
    status = "live"
    route = "/analysis"

    def __init__(self, vortex_service=None):
        """Optional vortex_service (VortexService) for /rotation/charts/vortex_model_* delegation."""
        self._vortex_service = vortex_service

    def validate(self, config):
        """Validate rotation curve request payload."""
        if not config:
            raise ValueError("Request body must be JSON")

        mass_model = config.get("mass_model")
        if not mass_model:
            raise ValueError("mass_model is required")

        mode = (config.get("mode") or "default").strip().lower()
        if mode not in ("default", "vortex"):
            mode = "default"

        return {
            "max_radius": config.get("max_radius", 30),
            "num_points": max(10, min(int(config.get("num_points", 100)), 500)),
            "accel_ratio": config.get("accel_ratio", 1.0),
            "mass_model": mass_model,
            "observations": config.get("observations"),
            "galactic_radius": config.get("galactic_radius"),
            "vortex_strength": config.get("vortex_strength"),
            "mode": mode,
        }

    def _cdm_halo(self, mass_model, observations, accel_ratio, galaxy_id=None):
        """Compute CDM halo info. For milky_way and m33 use literature NFW (letter fig1); else fit or abundance matching."""
        if galaxy_id and galaxy_id in LITERATURE_NFW_RHO0_RS:
            rho0, rs_m = LITERATURE_NFW_RHO0_RS[galaxy_id]
            m200 = m200_from_nfw_rho0_rs(rho0, rs_m)
            c_val = concentration(m200)
            r200 = r200_kpc(m200)
            cdm_halo_info = {
                "m200": round(m200, 2),
                "c": round(c_val, 2),
                "r200_kpc": round(r200, 2),
                "n_params_fitted": 0,
                "n_params_total": 2,
                "method": "literature NFW (letter fig1)",
            }
            return m200, cdm_halo_info

        m_total = total_mass(mass_model)

        def mass_at_r(r):
            return enclosed_mass(r, mass_model)

        cdm_halo_info = None
        m200 = None

        if observations and len(observations) >= 2:
            fit = fit_halo(observations, mass_at_r, accel_ratio)
            if fit:
                m200 = fit['m200']
                cdm_halo_info = fit
                cdm_halo_info['method'] = 'chi-squared fit to observations'

        if m200 is None:
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

        if cdm_halo_info:
            cdm_halo_info['m200'] = round(cdm_halo_info['m200'], 2)
        return m200, cdm_halo_info

    @staticmethod
    def _field_geometry(mass_model, accel_ratio):
        """Compute field geometry from mass model via topological yN
        conditions.  No observations or galactic_radius needed."""
        b = mass_model.get("bulge", {})
        d = mass_model.get("disk", {})
        g = mass_model.get("gas", {})
        a0_eff = constants.A0 * accel_ratio
        return solve_field_geometry(
            b.get("M", 0), b.get("a", 0),
            d.get("M", 0), d.get("Rd", 0),
            g.get("M", 0), g.get("Rd", 0),
            a0_eff,
        )

    def compute(self, config):
        """Compute rotation curves (prediction mode, no GA)."""
        mass_model = config["mass_model"]
        accel_ratio = config["accel_ratio"]
        observations = config.get("observations")

        m200, cdm_halo_info = self._cdm_halo(
            mass_model, observations, accel_ratio)

        # Compute theoretical Origin Throughput from gas leverage
        gr = config.get("galactic_radius")
        theoretical_ot = (
            auto_vortex_strength(mass_model, float(gr)) if gr else 0.0
        )

        # Use explicit override or theoretical prediction (never GA)
        vortex_val = config.get("vortex_strength")
        if vortex_val is None:
            vortex_val = theoretical_ot

        engine_config = GravisConfig(
            mass_model=mass_model,
            max_radius=config["max_radius"],
            num_points=config["num_points"],
            accel_ratio=accel_ratio,
            galactic_radius=config.get("galactic_radius"),
            vortex_strength=vortex_val,
        )
        engine = GravisEngine.rotation_curve(engine_config, m200=m200)
        result = engine.run()

        response = result.to_api_response()
        response["auto_origin_throughput"] = vortex_val
        response["theoretical_origin_throughput"] = theoretical_ot
        response["cdm_halo"] = cdm_halo_info

        observations = config.get("observations")
        r_first_obs = 0.0
        if observations:
            pos_r = [float(o.get("r", 0)) for o in observations if o.get("r") and float(o.get("r", 0)) > 0]
            if pos_r:
                r_first_obs = min(pos_r)

        if config.get("mode") == "vortex":
            radii = response["radii"]
            series_dict = {
                k: response[k] for k in ("newtonian", "dtg", "mond", "cdm")
                if k in response
            }
            if "enclosed_mass" in response:
                series_dict["enclosed_mass"] = response["enclosed_mass"]
            radii_sym, series_sym = mirror_curve_with_cutoff(
                radii, series_dict, r_first_obs, bridge_fv_percent=1.0)
            response["radii"] = radii_sym
            for k, vals in series_sym.items():
                response[k] = vals

        # Field geometry: always compute from mass model via topological
        # yN conditions. No observations or galactic_radius needed.
        response["field_geometry"] = self._field_geometry(
            mass_model, accel_ratio)

        # Fit quality metrics (always computed when observations exist)
        response["metrics"] = compute_fit_metrics(
            [float(r) for r in result.radii],
            [float(v) for v in result.series("gfd")],
            observations,
            engine_config,
        )

        resp_radii = response.get("radii", [])
        obs_r_vals = []
        obs_v_bounds = []
        if observations:
            for o in observations:
                rv = float(o.get("r", 0))
                vv = float(o.get("v", 0))
                ev = float(o.get("err", 0))
                if rv > 0:
                    obs_r_vals.append(rv)
                if vv > 0:
                    obs_v_bounds.append(vv + ev)
        max_obs_r = max(obs_r_vals) if obs_r_vals else 0
        x_max = max(
            max_obs_r * 1.15,
            (float(config.get("galactic_radius") or 0)) * 1.1,
        ) or config["max_radius"]
        all_v = list(obs_v_bounds)
        for key in ("dtg", "newtonian", "mond", "cdm"):
            vals = response.get(key, [])
            for i, v in enumerate(vals):
                if i < len(resp_radii) and resp_radii[i] <= x_max:
                    if v is not None and not math.isnan(v):
                        all_v.append(v)
        y_max_raw = max(all_v) if all_v else 50
        y_max = math.ceil(y_max_raw * 1.15 / 10) * 10
        response["axis_bounds"] = {
            "x_min": 0,
            "x_max": round(x_max, 2),
            "y_min": 0,
            "y_max": y_max,
        }

        return response

    def compute_infer(self, config):
        """Compute rotation curves with inference optimization.

        Stage 1: Accept published mass model as baseline.
        Stage 2: Grid search finds throughput for best GFD-sigma fit.
        Stage 3: Mass decomposition refines stellar masses.
        """
        mass_model = config["mass_model"]
        accel_ratio = config["accel_ratio"]
        observations = config.get("observations")
        gr = config.get("galactic_radius")

        theoretical_ot = (
            auto_vortex_strength(mass_model, float(gr)) if gr else 0.0
        )


        vortex_val = config.get("vortex_strength")
        infer_result = None
        run_mass_model = mass_model

        # After inference, use the derived R_env for the engine so the
        # displayed sigma curve and grid are consistent with the optimizer.
        engine_gr = gr
        if vortex_val is None:
            if observations and len(observations) >= 3:
                infer_result = optimize_inference(
                    mass_model,
                    config["max_radius"],
                    config["num_points"],
                    observations,
                    accel_ratio,
                    float(gr) if gr else 0.0,
                )
                vortex_val = infer_result["throughput"]
                run_mass_model = infer_result["mass_model"]
                # Use the topologically derived R_env for the engine
                fg = infer_result.get("field_geometry", {})
                if fg and fg.get("envelope_radius_kpc"):
                    engine_gr = fg["envelope_radius_kpc"]
            else:
                vortex_val = theoretical_ot

        m200, cdm_halo_info = self._cdm_halo(
            run_mass_model, observations, accel_ratio)

        engine_config = GravisConfig(
            mass_model=run_mass_model,
            max_radius=config["max_radius"],
            num_points=config["num_points"],
            accel_ratio=accel_ratio,
            galactic_radius=engine_gr,
            vortex_strength=vortex_val,
        )
        engine = GravisEngine.rotation_curve(engine_config, m200=m200)
        result = engine.run()

        response = result.to_api_response()
        response["auto_origin_throughput"] = vortex_val
        response["theoretical_origin_throughput"] = theoretical_ot

        if infer_result and infer_result.get("method", "").startswith("inference"):
            response["throughput_fit"] = {
                "method": infer_result["method"],
                "gfd_rms_km_s": infer_result["gfd_rms"],
                "rms_km_s": infer_result["rms"],
                "chi2_dof": infer_result["chi2_dof"],
            }
            response["optimized_mass_model"] = run_mass_model
            if "gene_report" in infer_result:
                response["gene_report"] = infer_result["gene_report"]
            if "band_coverage" in infer_result:
                response["band_coverage"] = infer_result["band_coverage"]

        # Field geometry: prefer inference result (uses Stage 3 fitted
        # masses), fall back to computing from current mass model.
        if infer_result and "field_geometry" in infer_result:
            response["field_geometry"] = infer_result["field_geometry"]
        else:
            response["field_geometry"] = self._field_geometry(
                run_mass_model, accel_ratio)

        response["cdm_halo"] = cdm_halo_info

        r_first_obs = 0.0
        if observations:
            pos_r = [float(o.get("r", 0)) for o in observations if o.get("r") and float(o.get("r", 0)) > 0]
            if pos_r:
                r_first_obs = min(pos_r)

        if config.get("mode") == "vortex":
            radii = response["radii"]
            series_dict = {
                k: response[k] for k in ("newtonian", "dtg", "mond", "cdm")
                if k in response
            }
            if "enclosed_mass" in response:
                series_dict["enclosed_mass"] = response["enclosed_mass"]
            radii_sym, series_sym = mirror_curve_with_cutoff(
                radii, series_dict, r_first_obs, bridge_fv_percent=1.0)
            response["radii"] = radii_sym
            for k, vals in series_sym.items():
                response[k] = vals
        else:
            if r_first_obs > 0:
                radii = response["radii"]
                series_dict = {
                    k: response[k] for k in ("newtonian", "dtg", "mond", "cdm")
                    if k in response
                }
                if "enclosed_mass" in response:
                    series_dict["enclosed_mass"] = response["enclosed_mass"]
                radii_tr, series_tr = truncate_to_first_obs(
                    radii, series_dict, r_first_obs)
                response["radii"] = radii_tr
                for k, vals in series_tr.items():
                    response[k] = vals

        # Fit quality metrics (computed against optimized mass model)
        response["metrics"] = compute_fit_metrics(
            [float(r) for r in result.radii],
            [float(v) for v in result.series("gfd")],
            observations,
            engine_config,
        )

        return response

    def compute_photometric_chart(self, galaxy_id, num_points=500,
                                 accel_ratio=1.0, max_radius=50.0, mode="default",
                                 gfd_velocity_smoothing_pct=6.2):
        """Unified photometric chart: one engine run, all four theory curves.

        Resolves photometric mass from catalog via derive_mass_parameters_from_photometry,
        runs GravisEngine once, returns radii and newtonian_photometric, gfd_photometric,
        mond_photometric, cdm_photometric, gfd_velocity (forward from mass via SST+Poisson), gfd_sst_velocity_decode (inverse then forward from observations only).
        No sandbox dependency for theory curves.
        """
        gal = get_galaxy_by_id(galaxy_id)
        if not gal:
            return None, {"error": "Galaxy not found: %s" % galaxy_id}, 404

        mass_model = gal.get("mass_model", {})
        if not mass_model:
            return None, {"error": "Galaxy has no mass model"}, 400

        a0_eff = constants.A0 * accel_ratio
        photometry = {
            "Mb": mass_model.get("bulge", {}).get("M", 0),
            "ab": mass_model.get("bulge", {}).get("a", 0),
            "Md": mass_model.get("disk", {}).get("M", 0),
            "Rd": mass_model.get("disk", {}).get("Rd", 0),
            "Mg": mass_model.get("gas", {}).get("M", 0),
            "Rg": mass_model.get("gas", {}).get("Rd", 0),
        }
        photo_result = derive_mass_parameters_from_photometry(photometry, a0_eff)
        if "error" in photo_result:
            return None, {"error": photo_result["error"]}, 400

        pm = photo_result["mass_model"]
        observations = gal.get("observations", [])
        m200, cdm_halo_info = self._cdm_halo(pm, observations, accel_ratio, galaxy_id=galaxy_id)

        gr = gal.get("galactic_radius")
        if gr is None:
            gr = max_radius
        gr = float(gr) if gr else max_radius
        obs_r_max_photo = 0.0
        for o in observations:
            rv = float(o.get("r", 0))
            if rv > obs_r_max_photo:
                obs_r_max_photo = rv
        engine_max_r_photo = max(max_radius, gr * 1.15, obs_r_max_photo * 1.25)
        theoretical_ot = auto_vortex_strength(pm, gr) if gr else 0.0

        engine_config = GravisConfig(
            mass_model=pm,
            max_radius=engine_max_r_photo,
            num_points=num_points,
            accel_ratio=accel_ratio,
            galactic_radius=gr,
            vortex_strength=theoretical_ot,
        )
        engine = GravisEngine.rotation_curve(engine_config, m200=m200)
        result = engine.run()

        radii = [round(r, 6) for r in result.radii]
        series_dict = {
            "newtonian_photometric": [round(float(v), 4) for v in result.series("newtonian")],
            "gfd_photometric": [round(float(v), 4) for v in result.series("gfd")],
            "mond_photometric": [round(float(v), 4) for v in result.series("mond")],
            "cdm_photometric": [round(float(v), 4) for v in result.series("cdm")],
            "gfd_velocity": [round(float(v), 4) for v in result.series("gfd_velocity")],
        }

        r_first_obs = 0.0
        if observations:
            pos_r = [float(o.get("r", 0)) for o in observations if o.get("r") and float(o.get("r", 0)) > 0]
            if pos_r:
                r_first_obs = min(pos_r)

        if mode == "vortex":
            radii, series_dict = mirror_curve_with_cutoff(
                radii, series_dict, r_first_obs, bridge_fv_percent=1.0)

        chart = {
            "radii": radii,
            "newtonian_photometric": series_dict.get("newtonian_photometric", []),
            "gfd_photometric": series_dict.get("gfd_photometric", []),
            "mond_photometric": series_dict.get("mond_photometric", []),
            "cdm_photometric": series_dict.get("cdm_photometric", []),
            "gfd_velocity": series_dict.get("gfd_velocity", []),
        }

        obs_r = []
        obs_v = []
        obs_v_bounds = []
        for o in observations:
            r = float(o.get("r", 0))
            v = float(o.get("v", 0))
            e = float(o.get("err", 0))
            if r > 0 and v > 0:
                obs_r.append(r)
                obs_v.append(v)
                obs_v_bounds.append(v + e)
        n_obs = len(obs_r)
        if n_obs >= 2:
            chart["gfd_sst_velocity_decode"] = [round(float(v), 4) for v in gfd_sst_velocity_decode_curve(
                obs_r, obs_v, radii, a0_eff)]
        else:
            chart["gfd_sst_velocity_decode"] = []

        r_hi = gal.get("galactic_radius", 0) or 0
        fg = photo_result.get("field_geometry", {})
        env_r = fg.get("envelope_radius_kpc", 0) or 0
        x_max_obs = max(
            env_r * 1.1,
            (max(obs_r) * 1.15) if obs_r else 0,
            r_hi * 1.1,
        ) or engine_max_r_photo
        all_v_obs = list(obs_v_bounds)
        for key in ("gfd_photometric", "newtonian_photometric",
                     "mond_photometric", "cdm_photometric",
                     "gfd_velocity"):
            vals = series_dict.get(key, [])
            for i, v in enumerate(vals):
                if v is not None and not math.isnan(v) and i < len(radii) and radii[i] <= x_max_obs:
                    all_v_obs.append(v)
        decode_vals = chart.get("gfd_sst_velocity_decode", [])
        for i, v in enumerate(decode_vals):
            if v is not None and not math.isnan(v) and i < len(radii) and radii[i] <= x_max_obs:
                all_v_obs.append(v)
        y_max_obs_raw = max(all_v_obs) if all_v_obs else 50
        y_max_obs = math.ceil(y_max_obs_raw * 1.15 / 10) * 10
        axis_bounds_obs = {
            "x_min": 0,
            "x_max": round(x_max_obs, 2),
            "y_min": 0,
            "y_max": y_max_obs,
        }

        residuals = []
        if n_obs > 0 and result.radii and result.series("gfd"):
            r_list = list(result.radii)
            gfd_list = list(result.series("gfd"))
            for o in observations:
                r = float(o.get("r", 0))
                v = float(o.get("v", 0))
                if r <= 0 or v <= 0:
                    continue
                vp = _interp(r_list, gfd_list, r)
                if vp is None:
                    continue
                err = max(float(o.get("err", 5.0)), 1.0)
                residuals.append({
                    "r": r, "v_obs": v, "v_gfd": round(vp, 2),
                    "delta": round(v - vp, 2), "err": err,
                })

        return {
            "galaxy": gal.get("name", galaxy_id),
            "chart": chart,
            "observations": observations,
            "axis_bounds": axis_bounds_obs,
            "photometric_mass_model": pm,
            "photometric_M_total": photo_result["M_total"],
            "field_geometry": photo_result["field_geometry"],
            "sparc_r_hi_kpc": gal.get("galactic_radius", 0),
            "cdm_halo": cdm_halo_info,
            "residuals": residuals,
            "n_obs": n_obs,
        }, None, None

    def compute_mass_model_chart(self, galaxy_id, num_points=500,
                                 accel_ratio=1.0, max_radius=50.0,
                                 mass_model_override=None):
        """Chart payload for Mass model tab only: radii, theory curves, observations. No gfd_velocity or gfd_sst_velocity_decode.
        When mass_model_override is provided, use it instead of the galaxy's photometric mass model (slider driven)."""
        gal = get_galaxy_by_id(galaxy_id)
        if not gal:
            return None, {"error": "Galaxy not found: %s" % galaxy_id}, 404
        mass_model = mass_model_override or gal.get("mass_model", {})
        if not mass_model:
            return None, {"error": "Galaxy has no mass model"}, 400
        a0_eff = constants.A0 * accel_ratio
        photometry = {
            "Mb": mass_model.get("bulge", {}).get("M", 0),
            "ab": mass_model.get("bulge", {}).get("a", 0),
            "Md": mass_model.get("disk", {}).get("M", 0),
            "Rd": mass_model.get("disk", {}).get("Rd", 0),
            "Mg": mass_model.get("gas", {}).get("M", 0),
            "Rg": mass_model.get("gas", {}).get("Rd", 0),
        }
        photo_result = derive_mass_parameters_from_photometry(photometry, a0_eff)
        if "error" in photo_result:
            return None, {"error": photo_result["error"]}, 400
        pm = photo_result["mass_model"]
        observations = gal.get("observations", [])
        m200, cdm_halo_info = self._cdm_halo(pm, observations, accel_ratio, galaxy_id=galaxy_id)
        gr = gal.get("galactic_radius")
        if gr is None:
            gr = max_radius
        gr = float(gr) if gr else max_radius
        obs_r_max = 0.0
        for o in observations:
            rv = float(o.get("r", 0))
            if rv > obs_r_max:
                obs_r_max = rv
        engine_max_r = max(max_radius, gr * 1.15, obs_r_max * 1.25)
        theoretical_ot = auto_vortex_strength(pm, gr) if gr else 0.0
        engine_config = GravisConfig(
            mass_model=pm,
            max_radius=engine_max_r,
            num_points=num_points,
            accel_ratio=accel_ratio,
            galactic_radius=gr,
            vortex_strength=theoretical_ot,
            observations=observations,
        )
        engine = GravisEngine.rotation_curve(engine_config, m200=m200)
        result = engine.run()
        radii = [round(r, 6) for r in result.radii]
        series_dict = {
            "newtonian_photometric": [round(float(v), 4) for v in result.series("newtonian")],
            "gfd_photometric": [round(float(v), 4) for v in result.series("gfd")],
            "mond_photometric": [round(float(v), 4) for v in result.series("mond")],
            "cdm_photometric": [round(float(v), 4) for v in result.series("cdm")],
        }
        if "gfd_topological" in result.stage_results:
            series_dict["gfd_topological"] = [round(float(v), 4) for v in result.series("gfd_topological")]
        chart = {
            "radii": radii,
            "newtonian_photometric": series_dict.get("newtonian_photometric", []),
            "gfd_photometric": series_dict.get("gfd_photometric", []),
            "mond_photometric": series_dict.get("mond_photometric", []),
            "cdm_photometric": series_dict.get("cdm_photometric", []),
        }
        if "gfd_topological" in series_dict:
            chart["gfd_topological"] = series_dict["gfd_topological"]

        obs_r_vals = []
        obs_v_vals = []
        for o in observations:
            rv = float(o.get("r", 0))
            vv = float(o.get("v", 0))
            ev = float(o.get("err", 0))
            if rv > 0:
                obs_r_vals.append(rv)
            if vv > 0:
                obs_v_vals.append(vv + ev)
                obs_v_vals.append(max(0, vv - ev))

        fg = photo_result.get("field_geometry", {})
        r_hi = gal.get("galactic_radius", 0) or 0
        env_r = fg.get("envelope_radius_kpc", 0) or 0
        max_obs_r = max(obs_r_vals) if obs_r_vals else 0
        x_max = max(
            env_r * 1.1,
            max_obs_r * 1.15,
            r_hi * 1.1,
        ) or max_radius
        all_v = list(obs_v_vals)
        for key in ("gfd_photometric", "newtonian_photometric",
                     "mond_photometric", "cdm_photometric",
                     "gfd_topological"):
            vals = series_dict.get(key, [])
            for i, v in enumerate(vals):
                if v is not None and not math.isnan(v) and radii[i] <= x_max:
                    all_v.append(v)
        y_min = 0
        y_max_raw = max(all_v) if all_v else 50
        y_max = math.ceil(y_max_raw * 1.15 / 10) * 10

        axis_bounds = {
            "x_min": 0,
            "x_max": round(x_max, 2),
            "y_min": y_min,
            "y_max": y_max,
        }

        return {
            "galaxy": gal.get("name", galaxy_id),
            "chart": chart,
            "observations": observations,
            "axis_bounds": axis_bounds,
            "photometric_mass_model": pm,
            "photometric_M_total": photo_result["M_total"],
            "field_geometry": photo_result["field_geometry"],
            "sparc_r_hi_kpc": gal.get("galactic_radius", 0),
            "cdm_halo": cdm_halo_info,
        }, None, None

    def compute_observation_model_chart(self, galaxy_id, num_points=500,
                                        accel_ratio=1.0, max_radius=50.0, mode="default",
                                        gfd_velocity_smoothing_pct=6.2):
        """Chart payload for Observations tab. Same as compute_photometric_chart (full chart including gfd_velocity, gfd_sst_velocity_decode)."""
        return self.compute_photometric_chart(
            galaxy_id, num_points=num_points, accel_ratio=accel_ratio,
            max_radius=max_radius, mode=mode,
            gfd_velocity_smoothing_pct=gfd_velocity_smoothing_pct)

    def compute_gfd_velocity_curve_v2(self, galaxy_id, num_points=500,
                                       accel_ratio=1.0, max_radius=50.0,
                                       deflection_smoothing_pct=6.2,
                                       include_inferred=True):
        """Velocity curve v2: GFD field velocity from mass profile (Eq 75 + SST). Same as panel script."""
        gal = get_galaxy_by_id(galaxy_id)
        if not gal:
            return None, {"error": "Galaxy not found: %s" % galaxy_id}, 404
        mass_model = gal.get("mass_model")
        if not mass_model:
            return None, {"error": "Galaxy has no mass model for v2 curve"}, 400
        observations = gal.get("observations", [])
        obs_r = [float(o["r"]) for o in observations if o.get("r") and float(o.get("r", 0)) > 0 and o.get("v") and float(o.get("v", 0)) > 0]
        obs_v = [float(o["v"]) for o in observations if o.get("r") and float(o.get("r", 0)) > 0 and o.get("v") and float(o.get("v", 0)) > 0]
        if obs_r:
            obs_r, obs_v = zip(*sorted(zip(obs_r, obs_v)))
            obs_r, obs_v = list(obs_r), list(obs_v)
        r_min = max(0.01, min(obs_r) * 0.5) if obs_r else 0.5
        r_max = min(max_radius, max(obs_r) * 1.2) if obs_r else float(max_radius)
        radii = [round(x, 6) for x in np.linspace(r_min, r_max, num_points).tolist()]
        a0_eff = constants.A0 * accel_ratio
        gfd_velocity_v2 = [round(float(v), 4) for v in gfd_velocity_curve_sst(
            radii, lambda r, mm=mass_model: enclosed_mass(r, mm), a0_eff)]

        out = {
            "galaxy": gal.get("name", galaxy_id),
            "radii": radii,
            "gfd_velocity_v2": gfd_velocity_v2,
            "n_obs": len(obs_r),
        }
        if include_inferred and gfd_velocity_v2:
            g_source_list = []
            M_enc_list = []
            for i, r in enumerate(radii):
                v_val = gfd_velocity_v2[i] if i < len(gfd_velocity_v2) else 0.0
                if v_val > 0 and r > 0:
                    g_source_list.append(round(
                        g_source_from_velocity(v_val, r, a0_eff), 6))
                    M_enc_list.append(round(
                        M_enc_from_velocity(v_val, r, a0_eff), 2))
                else:
                    g_source_list.append(0.0)
                    M_enc_list.append(0.0)
            out["g_source_ms2"] = g_source_list
            out["inferred_M_enc_solar"] = M_enc_list
        return out, None, None

    def register_routes(self, bp):
        """Mount all rotation-specific API endpoints."""
        service = self

        # -- Photometric chart (unified: all four theory curves from one engine run) --
        @bp.route("/rotation/photometric-curve", methods=["POST"])
        def rotation_photometric_curve():
            data = request.get_json()
            if not data:
                return jsonify({"error": "Request body must be JSON"}), 400
            galaxy_id = data.get("galaxy_id")
            if not galaxy_id:
                return jsonify({"error": "galaxy_id is required"}), 400
            num_points = max(10, min(int(data.get("num_points", 500)), 500))
            accel_ratio = float(data.get("accel_ratio", 1.0))
            max_radius = float(data.get("max_radius", 50.0))
            mode = (data.get("mode") or "default").strip().lower()
            if mode not in ("default", "vortex"):
                mode = "default"
            gfd_velocity_smoothing_pct = float(data.get("gfd_velocity_smoothing_pct", 6.2))
            result, err, status = service.compute_photometric_chart(
                galaxy_id, num_points=num_points, accel_ratio=accel_ratio,
                max_radius=max_radius, mode=mode,
                gfd_velocity_smoothing_pct=gfd_velocity_smoothing_pct)
            if err is not None:
                return jsonify(err), status if status else 400
            return jsonify(result)

        # -- Chart-specific endpoints (one per chart screen) --
        @bp.route("/rotation/charts/mass_model", methods=["POST"])
        def rotation_chart_mass_model():
            data = request.get_json()
            if not data:
                return jsonify({"error": "Request body must be JSON"}), 400
            galaxy_id = data.get("galaxy_id")
            if not galaxy_id:
                return jsonify({"error": "galaxy_id is required"}), 400
            num_points = max(10, min(int(data.get("num_points", 500)), 500))
            accel_ratio = float(data.get("accel_ratio", 1.0))
            max_radius = float(data.get("max_radius", 50.0))
            mass_model_override = data.get("mass_model_override")
            result, err, status = service.compute_mass_model_chart(
                galaxy_id, num_points=num_points, accel_ratio=accel_ratio,
                max_radius=max_radius,
                mass_model_override=mass_model_override)
            if err is not None:
                return jsonify(err), status if status else 400
            return jsonify(result)

        @bp.route("/rotation/charts/observation_model", methods=["POST"])
        def rotation_chart_observation_model():
            data = request.get_json()
            if not data:
                return jsonify({"error": "Request body must be JSON"}), 400
            galaxy_id = data.get("galaxy_id")
            if not galaxy_id:
                return jsonify({"error": "galaxy_id is required"}), 400
            num_points = max(10, min(int(data.get("num_points", 500)), 500))
            accel_ratio = float(data.get("accel_ratio", 1.0))
            max_radius = float(data.get("max_radius", 50.0))
            result, err, status = service.compute_observation_model_chart(
                galaxy_id, num_points=num_points, accel_ratio=accel_ratio,
                max_radius=max_radius)
            if err is not None:
                return jsonify(err), status if status else 400
            return jsonify(result)

        if service._vortex_service is not None:
            @bp.route("/rotation/charts/vortex_model_a", methods=["POST"])
            def rotation_chart_vortex_model_a():
                data = request.get_json()
                if not data:
                    return jsonify({"error": "Request body must be JSON"}), 400
                galaxy_id = data.get("galaxy_id")
                if not galaxy_id:
                    return jsonify({"error": "galaxy_id is required"}), 400
                variance_pct = float(data.get("variance_pct", 1.5))
                out = service._vortex_service.figure_a(galaxy_id.strip().lower(), variance_pct)
                if not out:
                    return jsonify({"error": "Galaxy not found or too few observations"}), 404
                return jsonify({"figure_a": out["figure_a"], "galaxy_id": out["galaxy_id"]})

            @bp.route("/rotation/charts/vortex_model_b", methods=["POST"])
            def rotation_chart_vortex_model_b():
                data = request.get_json()
                if not data:
                    return jsonify({"error": "Request body must be JSON"}), 400
                galaxy_id = data.get("galaxy_id")
                if not galaxy_id:
                    return jsonify({"error": "galaxy_id is required"}), 400
                variance_pct = float(data.get("variance_pct", 1.5))
                out = service._vortex_service.figure_b(galaxy_id.strip().lower(), variance_pct)
                if not out:
                    return jsonify({"error": "Galaxy not found or too few observations"}), 404
                return jsonify({"figure_b": out["figure_b"], "galaxy_id": out["galaxy_id"]})

        # -- GFD velocity curve v2: deflection = direct SST field reading (no R_t, R_env) --
        @bp.route("/rotation/gfd-velocity-curve-v2", methods=["POST"])
        def rotation_gfd_velocity_curve_v2():
            data = request.get_json()
            if not data:
                return jsonify({"error": "Request body must be JSON"}), 400
            galaxy_id = data.get("galaxy_id")
            if not galaxy_id:
                return jsonify({"error": "galaxy_id is required"}), 400
            num_points = max(10, min(int(data.get("num_points", 500)), 500))
            accel_ratio = float(data.get("accel_ratio", 1.0))
            max_radius = float(data.get("max_radius", 50.0))
            deflection_smoothing_pct = float(data.get("deflection_smoothing_pct", 6.2))
            include_inferred = data.get("include_inferred", True)
            result, err, status = service.compute_gfd_velocity_curve_v2(
                galaxy_id, num_points=num_points, accel_ratio=accel_ratio,
                max_radius=max_radius, deflection_smoothing_pct=deflection_smoothing_pct,
                include_inferred=include_inferred)
            if err is not None:
                return jsonify(err), status if status else 400
            return jsonify(result)

        # -- Rotation curve computation --
        @bp.route("/rotation/curve", methods=["POST"])
        def rotation_curve():
            data = request.get_json()
            try:
                config = service.validate(data)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            result = service.compute(config)
            return jsonify(result)

        # -- Inference rotation curve --
        @bp.route("/rotation/inference", methods=["POST"])
        @bp.route("/rotation/infer-curve", methods=["POST"])
        def rotation_inference():
            data = request.get_json()
            try:
                config = service.validate(data)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            result = service.compute_infer(config)
            return jsonify(result)

        # -- Single-point mass inference --
        @bp.route("/rotation/infer-mass", methods=["POST"])
        def rotation_infer_mass():
            data = request.get_json()
            if not data:
                return jsonify({"error": "Request body must be JSON"}), 400

            r_kpc = data.get("r_kpc", 8.0)
            v_km_s = data.get("v_km_s", 230.0)
            accel_ratio = data.get("accel_ratio", 1.0)

            result = GravisEngine.infer_mass(r_kpc, v_km_s, accel_ratio)
            mass = result.series[0]

            return jsonify({
                "inferred_mass_solar": mass,
                "log10_mass": round(math.log10(max(mass, 1.0)), 4),
            })

        # -- Infer scaled mass model --
        @bp.route("/rotation/infer-mass-model", methods=["POST"])
        def rotation_infer_mass_model():
            data = request.get_json()
            if not data:
                return jsonify({"error": "Request body must be JSON"}), 400

            r_kpc = data.get("r_kpc", 8.0)
            v_km_s = data.get("v_km_s", 230.0)
            accel_ratio = data.get("accel_ratio", 1.0)
            mass_model = data.get("mass_model")

            if not mass_model:
                return jsonify({"error": "mass_model is required"}), 400

            m_enclosed_needed = infer_mass(r_kpc, v_km_s, accel_ratio)
            m_enclosed_current = enclosed_mass(r_kpc, mass_model)
            m_total_current = total_mass(mass_model)

            if m_enclosed_current <= 0 or m_total_current <= 0:
                return jsonify({
                    "error": "Mass model has zero enclosed mass "
                             "at observation radius"
                }), 400

            scale = m_enclosed_needed / m_enclosed_current

            scaled_model = {}
            for comp in ("bulge", "disk", "gas"):
                if comp in mass_model and mass_model[comp].get("M", 0) > 0:
                    entry = dict(mass_model[comp])
                    entry["M"] = entry["M"] * scale
                    scaled_model[comp] = entry
                elif comp in mass_model:
                    scaled_model[comp] = dict(mass_model[comp])

            inferred_total = total_mass(scaled_model)

            v_ms = v_km_s * 1000.0
            a0_eff = constants.A0 * accel_ratio
            btfr_mass = (v_ms ** 4) / (constants.G * a0_eff) / constants.M_SUN

            return jsonify({
                "inferred_enclosed": round(m_enclosed_needed, 2),
                "scale_factor": round(scale, 6),
                "inferred_mass_model": scaled_model,
                "inferred_total": round(inferred_total, 2),
                "log10_total": round(
                    math.log10(max(inferred_total, 1.0)), 4),
                "btfr_mass": round(btfr_mass, 2),
                "log10_btfr": round(
                    math.log10(max(btfr_mass, 1.0)), 4),
            })

        # -- Multi-point inference --
        @bp.route("/rotation/infer-mass-multi", methods=["POST"])
        def rotation_infer_mass_multi():
            data = request.get_json()
            if not data:
                return jsonify({"error": "Request body must be JSON"}), 400

            observations = data.get("observations", [])
            accel_ratio = data.get("accel_ratio", 1.0)
            mass_model = data.get("mass_model")

            if not mass_model or not observations:
                return jsonify({
                    "error": "mass_model and observations are required"
                }), 400

            results = []
            totals = []
            weights = []

            m_total_current = total_mass(mass_model)
            if m_total_current <= 0:
                return jsonify({
                    "error": "Mass model has zero total mass"
                }), 400

            mass_errors = []

            for obs in observations:
                r_kpc = obs.get("r", 0)
                v_km_s = obs.get("v", 0)
                err = obs.get("err", 0)
                if r_kpc <= 0 or v_km_s <= 0:
                    continue

                m_enclosed_needed = infer_mass(r_kpc, v_km_s, accel_ratio)
                m_enclosed_current = enclosed_mass(r_kpc, mass_model)

                if m_enclosed_current <= 0:
                    continue

                scale = m_enclosed_needed / m_enclosed_current
                inferred_total = m_total_current * scale
                enc_frac = m_enclosed_current / m_total_current

                sigma_err = max(err, 1.0)
                m_hi = infer_mass(
                    r_kpc, v_km_s + sigma_err, accel_ratio)
                m_lo = infer_mass(
                    r_kpc, max(v_km_s - sigma_err, 1.0), accel_ratio)
                scale_hi = m_hi / m_enclosed_current
                scale_lo = m_lo / m_enclosed_current
                total_hi = m_total_current * scale_hi
                total_lo = m_total_current * scale_lo
                delta_m = abs(total_hi - total_lo) / 2.0

                v_gfd = dtg_velocity(r_kpc, m_enclosed_current, accel_ratio)
                delta_v = round(v_km_s - v_gfd, 2)
                sigma_dev = (
                    round(abs(delta_v) / max(err, 0.1), 2)
                    if err > 0 else None
                )

                results.append({
                    "r_kpc": r_kpc,
                    "v_km_s": v_km_s,
                    "err": err,
                    "inferred_total": round(inferred_total, 2),
                    "log10_total": round(
                        math.log10(max(inferred_total, 1.0)), 4),
                    "enclosed_frac": round(enc_frac, 4),
                    "v_gfd": round(v_gfd, 2),
                    "delta_v": delta_v,
                    "sigma_dev": sigma_dev,
                })
                totals.append(inferred_total)
                weights.append(enc_frac)
                mass_errors.append(delta_m)

            if len(totals) < 2:
                return jsonify({
                    "error": "Need at least 2 valid observation points"
                }), 400

            n = len(totals)

            mean_total = sum(totals) / n
            variance = sum(
                (t - mean_total) ** 2 for t in totals) / (n - 1)
            std_total = math.sqrt(variance)
            cv = (
                (std_total / mean_total) * 100.0
                if mean_total > 0 else 0.0
            )

            w_sum = sum(weights)
            if w_sum > 0:
                w_mean = sum(
                    t * w for t, w in zip(totals, weights)) / w_sum
                w_sum2 = sum(w * w for w in weights)
                if w_sum * w_sum - w_sum2 > 0:
                    w_var = (
                        (w_sum / (w_sum * w_sum - w_sum2))
                        * sum(w * (t - w_mean) ** 2
                              for t, w in zip(totals, weights))
                    )
                    w_std = math.sqrt(max(w_var, 0))
                else:
                    w_std = 0.0
                w_cv = (
                    (w_std / w_mean) * 100.0 if w_mean > 0 else 0.0
                )
            else:
                w_mean = mean_total
                w_std = std_total
                w_cv = cv

            sorted_totals = sorted(totals)

            band_scatter = w_std

            if w_sum > 0 and mass_errors:
                w_err_sq = sum(
                    w * de * de
                    for w, de in zip(weights, mass_errors)
                )
                band_obs_err = math.sqrt(w_err_sq / w_sum)
            else:
                band_obs_err = std_total

            def _percentile(data, p):
                k = (len(data) - 1) * p
                f = int(k)
                c = min(f + 1, len(data) - 1)
                d = k - f
                return data[f] + d * (data[c] - data[f])

            q1 = _percentile(sorted_totals, 0.25)
            q3 = _percentile(sorted_totals, 0.75)
            band_iqr = (q3 - q1) / 2.0

            diag_pts = [
                p for p in results
                if p.get("enclosed_frac", 0) >= 0.05
                and p.get("delta_v") is not None
                and p.get("sigma_dev") is not None
            ]
            shape_diagnostic = None
            if len(diag_pts) >= 4:
                mid = len(diag_pts) // 2
                inner = diag_pts[:mid]
                outer = diag_pts[mid:]
                shape_diagnostic = {
                    "inner_r_max": inner[-1]["r_kpc"],
                    "outer_r_min": outer[0]["r_kpc"],
                    "inner_mean_dv": round(
                        sum(p["delta_v"] for p in inner)
                        / len(inner), 2),
                    "outer_mean_dv": round(
                        sum(p["delta_v"] for p in outer)
                        / len(outer), 2),
                    "inner_mean_sigma": round(
                        sum(p["sigma_dev"] for p in inner)
                        / len(inner), 2),
                    "outer_mean_sigma": round(
                        sum(p["sigma_dev"] for p in outer)
                        / len(outer), 2),
                    "n_inner": len(inner),
                    "n_outer": len(outer),
                }

            return jsonify({
                "points": results,
                "n_points": n,
                "mean_total": round(mean_total, 2),
                "std_total": round(std_total, 2),
                "log10_mean": round(
                    math.log10(max(mean_total, 1.0)), 4),
                "cv_percent": round(cv, 2),
                "weighted_mean": round(w_mean, 2),
                "weighted_std": round(w_std, 2),
                "log10_weighted_mean": round(
                    math.log10(max(w_mean, 1.0)), 4),
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

        # -- Field Analysis (GFD telemetry) --
        @bp.route("/rotation/field_analysis", methods=["POST"])
        def rotation_field_analysis():
            data = request.get_json()
            if not data:
                return jsonify({"error": "Request body must be JSON"}), 400

            mass_model = data.get("mass_model")
            galactic_radius = data.get("galactic_radius")
            vortex_strength = data.get("vortex_strength")
            accel_ratio = data.get("accel_ratio", 1.0)

            if not mass_model or not galactic_radius or vortex_strength is None:
                return jsonify({
                    "error": "mass_model, galactic_radius, and "
                             "vortex_strength are required"
                }), 400

            result = compute_field_analysis(
                mass_model,
                float(galactic_radius),
                float(vortex_strength),
                accel_ratio=float(accel_ratio),
            )
            return jsonify(result)

        # -- Galaxy catalog (data layer: data.galaxies; includes built-in + sparc/) --
        @bp.route("/rotation/galaxies", methods=["GET"])
        def rotation_list_galaxies():
            return jsonify(get_all_galaxies())

        @bp.route("/rotation/galaxy-ids", methods=["GET"])
        def rotation_list_galaxy_ids():
            """Return all galaxy ids we have (built-in examples + sparc folder)."""
            all_galaxies = get_all_galaxies()
            ids = []
            for mode in ("prediction", "inference"):
                for g in all_galaxies.get(mode) or []:
                    if isinstance(g, dict) and g.get("id"):
                        ids.append(g["id"])
            return jsonify({"ids": ids})

        @bp.route("/rotation/galaxies/<galaxy_id>", methods=["GET"])
        def rotation_get_galaxy(galaxy_id):
            galaxy = get_galaxy_by_id(galaxy_id)
            if galaxy is None:
                return jsonify({"error": "Galaxy not found"}), 404
            return jsonify(galaxy)

        @bp.route("/rotation/galaxy-catalog", methods=["GET"])
        def rotation_galaxy_catalog():
            """Lightweight list of {id, name} for all galaxies in sparc/."""
            return jsonify(get_galaxy_catalog())
