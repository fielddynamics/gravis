"""
Rotation Curves Service for GRAVIS.

Implements the GravisService interface for galactic rotation curve
analysis. This service wraps the existing GravisEngine pipeline and
owns all rotation-specific API endpoints under /api/rotation/*.

Endpoints:
    POST /api/rotation/curve           - compute rotation curves
    POST /api/rotation/infer-mass      - single-point mass inference
    POST /api/rotation/infer-mass-model - infer scaled mass model
    POST /api/rotation/infer-mass-multi - multi-point inference
    GET  /api/rotation/galaxies        - list galaxies
    GET  /api/rotation/galaxies/<id>   - get single galaxy

IMPORTANT: No unicode characters allowed (Windows charmap constraint).
"""

import math

from flask import jsonify, request

from physics.services import GravisService
from physics.engine import GravisConfig, GravisEngine
from physics import constants
from physics.mass_model import enclosed_mass, total_mass
from physics.inference import infer_mass
from physics.aqual import velocity as dtg_velocity
from physics.nfw import (
    abundance_matching, fit_halo, concentration, r200_kpc,
)
from data.galaxies import get_all_galaxies, get_galaxy_by_id


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

    def validate(self, config):
        """Validate rotation curve request payload."""
        if not config:
            raise ValueError("Request body must be JSON")

        mass_model = config.get("mass_model")
        if not mass_model:
            raise ValueError("mass_model is required")

        return {
            "max_radius": config.get("max_radius", 30),
            "num_points": max(10, min(int(config.get("num_points", 100)), 500)),
            "accel_ratio": config.get("accel_ratio", 1.0),
            "mass_model": mass_model,
            "observations": config.get("observations"),
            "galactic_radius": config.get("galactic_radius"),
        }

    def compute(self, config):
        """Compute rotation curves via the GravisEngine pipeline."""
        mass_model = config["mass_model"]
        accel_ratio = config["accel_ratio"]
        observations = config.get("observations")

        # Determine NFW halo mass
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

        # Run engine pipeline
        engine_config = GravisConfig(
            mass_model=mass_model,
            max_radius=config["max_radius"],
            num_points=config["num_points"],
            accel_ratio=accel_ratio,
            galactic_radius=config.get("galactic_radius"),
        )
        engine = GravisEngine.rotation_curve(engine_config, m200=m200)
        result = engine.run()

        response = result.to_api_response()

        if cdm_halo_info:
            cdm_halo_info['m200'] = round(cdm_halo_info['m200'], 2)
        response["cdm_halo"] = cdm_halo_info

        return response

    def register_routes(self, bp):
        """Mount all rotation-specific API endpoints."""
        service = self

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

        # -- Galaxy catalog --
        @bp.route("/rotation/galaxies", methods=["GET"])
        def rotation_list_galaxies():
            return jsonify(get_all_galaxies())

        @bp.route("/rotation/galaxies/<galaxy_id>", methods=["GET"])
        def rotation_get_galaxy(galaxy_id):
            galaxy = get_galaxy_by_id(galaxy_id)
            if galaxy is None:
                return jsonify({"error": "Galaxy not found"}), 404
            return jsonify(galaxy)
