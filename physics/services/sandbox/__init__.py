"""
Inference Sandbox Service for GRAVIS.

Proves that the complete galactic field model can be derived from
observation data alone, with zero dependency on published masses or
user-provided galactic_radius.

Endpoints:
    POST /api/sandbox/infer               - pure observation-driven inference
    POST /api/sandbox/compare             - side-by-side comparison
    POST /api/sandbox/map_gfd_with_bayesian - Bayesian GFD base fit

IMPORTANT: No unicode characters allowed (Windows charmap constraint).
"""

from flask import jsonify, request

from physics.services import GravisService
from physics.services.sandbox.pure_inference import infer_from_observations
import math
from physics.services.sandbox.bayesian_fit import (
    fit_gfd_bayesian,
    derive_mass_parameters_from_photometry,
    fit_observation_with_spline_then_gfd_bayesian,
    fit_observation_with_spline_then_gfd_bayesian_with_acceleration,
    gfd_velocity,
    gfd_velocity_curve,
)
from physics.services.rotation.inference import optimize_inference, solve_field_geometry
from physics.services.rotation.vortex_bridge import (
    mirror_curve_with_cutoff,
    truncate_to_first_obs,
)
from physics.constants import A0, KPC_TO_M
from data.galaxies import get_all_galaxies, get_galaxy_by_id


class InferenceSandboxService(GravisService):
    """
    Sandbox service for testing pure observation-driven inference.

    This service implements the complete chain:
      observations -> invert field equation -> mass model ->
      throat (y_N = 18/65) -> horizon (R_env = R_t / 0.30) ->
      sigma stage -> throughput -> refined masses

    No published masses. No user-provided galactic_radius.
    Observations are the only input.
    """

    id = "sandbox"
    name = "Inference Sandbox"
    description = "Pure observation-driven inference (no published masses)"
    category = "galactic"
    status = "live"
    route = "/inference"

    def validate(self, config):
        """Validate sandbox request payload."""
        if not config:
            raise ValueError("Request body must be JSON")
        observations = config.get("observations")
        if not observations or len(observations) < 3:
            raise ValueError("At least 3 observations required")
        return {
            "observations": observations,
            "accel_ratio": float(config.get("accel_ratio", 1.0)),
            "num_points": max(50, min(
                int(config.get("num_points", 500)), 1000)),
        }

    def compute(self, config):
        """Run pure observation-driven inference."""
        return infer_from_observations(
            config["observations"],
            config["accel_ratio"],
            config["num_points"],
        )

    def register_routes(self, bp):
        """Mount sandbox API endpoints."""
        service = self

        @bp.route("/sandbox/infer", methods=["POST"])
        def sandbox_infer():
            """Pure observation-driven inference.

            Input JSON:
                observations: [{r, v, err}, ...]
                accel_ratio: float (default 1.0)

            Returns full derived model: masses, geometry, throughput.
            """
            data = request.get_json()
            try:
                config = service.validate(data)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400

            result = service.compute(config)
            return jsonify(result)

        @bp.route("/sandbox/compare", methods=["POST"])
        def sandbox_compare():
            """Compare pure pipeline vs existing pipeline.

            Input JSON:
                galaxy_id: str (from catalog)
                  OR
                observations: [{r, v, err}, ...]
                mass_model: {bulge, disk, gas}
                galactic_radius: float

            Returns side-by-side comparison.
            """
            data = request.get_json()
            if not data:
                return jsonify({"error": "Request body required"}), 400

            galaxy_id = data.get("galaxy_id")
            if galaxy_id:
                gal = get_galaxy_by_id(galaxy_id)
                if not gal:
                    return jsonify({
                        "error": "Galaxy not found: %s" % galaxy_id
                    }), 404
                observations = gal.get("observations", [])
                mass_model = gal.get("mass_model", {})
                galactic_radius = gal.get("galactic_radius", 0)
                accel_ratio = gal.get("accel", 1.0)
                name = gal.get("name", galaxy_id)
                max_radius = gal.get("distance", 30)
            else:
                observations = data.get("observations", [])
                mass_model = data.get("mass_model", {})
                galactic_radius = data.get("galactic_radius", 0)
                accel_ratio = float(data.get("accel_ratio", 1.0))
                name = data.get("name", "custom")
                max_radius = float(data.get("max_radius", 30))

            if not observations or len(observations) < 3:
                return jsonify({
                    "error": "At least 3 observations required"
                }), 400

            num_points = int(data.get("num_points", 500))

            # Run pure observation-driven pipeline
            pure_result = infer_from_observations(
                observations, accel_ratio, num_points)

            # Run existing pipeline (uses published masses + R_env)
            existing_result = None
            if mass_model and galactic_radius:
                try:
                    existing_result = optimize_inference(
                        mass_model, max_radius, num_points,
                        observations, accel_ratio,
                        float(galactic_radius))
                except Exception as e:
                    existing_result = {"error": str(e)}

            return jsonify({
                "galaxy": name,
                "n_observations": len(observations),
                "pure_observation": pure_result,
                "existing_pipeline": existing_result,
            })

        @bp.route("/sandbox/photometric", methods=["POST"])
        def sandbox_photometric():
            """Photometric GFD curve with optional observation fits.

            Input JSON:
                galaxy_id: str (from catalog), required unless mode='manual'
                num_points: int (default 500)
                mode: str (default 'mass_model')
                    'mass_model' - photometric curve + field geometry only (instant)
                    'observations' - adds GFD Sigma + GFD Accel fits (~2-3s)
                    'manual' - GFD from provided mass_model (no galaxy_id)
                mass_model: dict (required if mode='manual') with bulge, disk, gas
                    each: M (solar), a or Rd (kpc)
            """
            data = request.get_json()
            if not data:
                return jsonify({"error": "Request body required"}), 400

            mode = data.get("mode", "mass_model")
            num_points = int(data.get("num_points", 500))
            accel_ratio = float(data.get("accel_ratio", 1.0))
            a0_eff = A0 * accel_ratio

            if mode == "manual":
                mass_model = data.get("mass_model")
                if not mass_model:
                    return jsonify({"error": "mass_model required for mode=manual"}), 400
                max_radius = float(data.get("max_radius", 50.0))
                Mb_p = mass_model.get("bulge", {}).get("M", 0)
                ab_p = mass_model.get("bulge", {}).get("a", 0)
                Md_p = mass_model.get("disk", {}).get("M", 0)
                Rd_p = mass_model.get("disk", {}).get("Rd", 0)
                Mg_p = mass_model.get("gas", {}).get("M", 0)
                Rg_p = mass_model.get("gas", {}).get("Rd", 0)
                photo_geom = solve_field_geometry(
                    Mb_p, ab_p, Md_p, Rd_p, Mg_p, Rg_p, a0_eff)
                r_env = photo_geom.get("envelope_radius_kpc") or 0
                chart_max = min(max_radius, max(r_env * 1.2, 100.0))
                dr = chart_max / num_points
                chart_radii = [dr * (i + 1) for i in range(num_points)]
                photo_vels = gfd_velocity_curve(
                    chart_radii,
                    Mb_p, ab_p, Md_p, Rd_p, Mg_p, Rg_p,
                    a0_eff)
                chart_out = {
                    "radii": [round(r, 4) for r in chart_radii],
                    "gfd_photometric": [round(v, 4) for v in photo_vels],
                }
                if data.get("chart_mode") == "vortex":
                    radii_sym, series_sym = mirror_curve_symmetric(
                        chart_out["radii"], {"gfd_photometric": chart_out["gfd_photometric"]})
                    chart_out["radii"] = radii_sym
                    chart_out["gfd_photometric"] = series_sym["gfd_photometric"]
                return jsonify({
                    "chart": chart_out,
                    "field_geometry": photo_geom,
                    "sparc_r_hi_kpc": 0,
                    "mode": "manual",
                })

            galaxy_id = data.get("galaxy_id")
            if not galaxy_id:
                return jsonify({
                    "error": "galaxy_id required"
                }), 400

            gal = get_galaxy_by_id(galaxy_id)
            if not gal:
                return jsonify({
                    "error": "Galaxy not found: %s" % galaxy_id
                }), 404

            mass_model = gal.get("mass_model", {})
            if not mass_model:
                return jsonify({
                    "error": "Galaxy has no mass model"
                }), 400

            observations = gal.get("observations", [])
            accel_ratio = gal.get("accel", 1.0)
            galaxy_name = gal.get("name", galaxy_id)
            a0_eff = A0 * accel_ratio
            num_points = int(data.get("num_points", 500))
            mode = data.get("mode", "mass_model")

            photometry = {
                "Mb": mass_model.get("bulge", {}).get("M", 0),
                "ab": mass_model.get("bulge", {}).get("a", 0),
                "Md": mass_model.get("disk", {}).get("M", 0),
                "Rd": mass_model.get("disk", {}).get("Rd", 0),
                "Mg": mass_model.get("gas", {}).get("M", 0),
                "Rg": mass_model.get("gas", {}).get("Rd", 0),
            }
            photo_result = derive_mass_parameters_from_photometry(
                photometry, a0_eff)
            if "error" in photo_result:
                return jsonify({
                    "error": photo_result["error"]
                }), 400

            pm = photo_result["mass_model"]
            Mb_p = pm["bulge"]["M"]
            ab_p = pm["bulge"]["a"]
            Md_p = pm["disk"]["M"]
            Rd_p = pm["disk"]["Rd"]
            Mg_p = pm["gas"]["M"]
            Rg_p = pm["gas"]["Rd"]

            photo_geom = solve_field_geometry(
                Mb_p, ab_p, Md_p, Rd_p, Mg_p, Rg_p, a0_eff)
            r_env = photo_geom.get("envelope_radius_kpc") or 0

            obs_r = []
            obs_v = []
            obs_err = []
            for o in observations:
                r = float(o.get("r", 0))
                v = float(o.get("v", 0))
                if r > 0 and v > 0:
                    obs_r.append(r)
                    obs_v.append(v)
                    obs_err.append(max(float(o.get("err", 5.0)), 1.0))

            max_obs_r = max(obs_r) if obs_r else 0
            r_vis_99 = photo_geom.get("visible_radius_99_kpc") or 0
            chart_max = max(
                r_env * 1.10, r_vis_99 * 1.10, max_obs_r * 1.15)
            dr = chart_max / num_points
            chart_radii = [dr * (i + 1) for i in range(num_points)]

            photo_vels = gfd_velocity_curve(
                chart_radii,
                Mb_p, ab_p, Md_p, Rd_p, Mg_p, Rg_p,
                a0_eff)

            residuals = []
            for j in range(len(obs_r)):
                vp = gfd_velocity(
                    obs_r[j],
                    Mb_p, ab_p, Md_p, Rd_p, Mg_p, Rg_p,
                    a0_eff)
                residuals.append({
                    "r": obs_r[j],
                    "v_obs": obs_v[j],
                    "v_gfd": round(vp, 2),
                    "delta": round(obs_v[j] - vp, 2),
                    "err": obs_err[j],
                })

            sparc_r_hi = gal.get("galactic_radius", 0)
            r_first_obs = min(obs_r) if obs_r else 0.0

            chart_out = {
                "radii": [round(r, 4) for r in chart_radii],
                "gfd_photometric": [round(v, 4) for v in photo_vels],
            }
            resp = {
                "galaxy": galaxy_name,
                "photometric_mass_model": pm,
                "photometric_M_total": photo_result["M_total"],
                "field_geometry": photo_geom,
                "sparc_r_hi_kpc": sparc_r_hi,
                "chart": chart_out,
                "residuals": residuals,
                "n_obs": len(obs_r),
                "mode": mode,
            }

            if mode == "observations":
                photo_p = {
                    "Mb": Mb_p, "ab": ab_p,
                    "Md": Md_p, "Rd": Rd_p,
                    "Mg": Mg_p, "Rg": Rg_p,
                }

                spline_result = (
                    fit_observation_with_spline_then_gfd_bayesian(
                        obs_r, obs_v, obs_err,
                        chart_radii, photo_vels, a0_eff,
                        photo_params=photo_p))

                if "error" not in spline_result:
                    resp["chart"]["gfd_spline"] = spline_result[
                        "spline_vels"]
                    resp["chart"]["gfd_covariant_spline"] = (
                        spline_result["gfd_covariant_spline"])
                    resp["chart"]["delta_v2_spline"] = (
                        spline_result["delta_v2"])
                    resp["vortex_signal_spline"] = spline_result[
                        "vortex_signal"]
                    resp["spline_rms"] = spline_result["rms"]

                accel_result = (
                    fit_observation_with_spline_then_gfd_bayesian_with_acceleration(
                        obs_r, obs_v, obs_err,
                        chart_radii, photo_vels, a0_eff,
                        photo_params=photo_p))

                if "error" not in accel_result:
                    resp["chart"]["gfd_accel"] = accel_result[
                        "accel_vels"]
                    resp["accel_ratio_fitted"] = accel_result[
                        "accel_ratio"]
                    resp["accel_rms"] = accel_result["rms"]
                    resp["vortex_signal_accel"] = accel_result[
                        "vortex_signal"]

            if data.get("chart_mode") == "vortex":
                radii_full = resp["chart"]["radii"]
                max_radius = data.get("max_radius")
                if max_radius is not None and float(max_radius) > 0:
                    max_radius = float(max_radius)
                    mask = [i for i, r in enumerate(radii_full) if float(r) <= max_radius]
                    if mask:
                        radii_full = [radii_full[i] for i in mask]
                        for k in list(resp["chart"].keys()):
                            if k != "radii" and isinstance(resp["chart"][k], list):
                                if len(resp["chart"][k]) == len(resp["chart"]["radii"]):
                                    resp["chart"][k] = [resp["chart"][k][i] for i in mask]
                        resp["chart"]["radii"] = radii_full
                series_full = {k: v for k, v in resp["chart"].items()
                               if k != "radii" and isinstance(v, list)
                               and len(v) == len(radii_full)}
                if series_full:
                    radii_sym, series_sym = mirror_curve_with_cutoff(
                        radii_full, series_full, r_first_obs, bridge_fv_percent=1.0)
                    resp["chart"]["radii"] = radii_sym
                    for k, vals in series_sym.items():
                        resp["chart"][k] = vals
            elif r_first_obs > 0:
                radii = resp["chart"]["radii"]
                series_dict = {k: v for k, v in resp["chart"].items()
                               if k != "radii" and isinstance(v, list)
                               and len(v) == len(radii)}
                if series_dict:
                    radii_tr, series_tr = truncate_to_first_obs(
                        radii, series_dict, r_first_obs)
                    resp["chart"]["radii"] = radii_tr
                    for k, vals in series_tr.items():
                        resp["chart"][k] = vals

            return jsonify(resp)

        @bp.route("/sandbox/map_gfd_with_bayesian", methods=["POST"])
        def sandbox_bayesian():
            """Bayesian GFD base fit to observations.

            Input JSON:
                galaxy_id: str (from catalog)
                  OR
                observations: [{r, v, err}, ...]
                accel_ratio: float (default 1.0)

            Returns fitted mass model, GFD base curve, residuals,
            field geometry, and chart data for plotting.
            """
            data = request.get_json()
            if not data:
                return jsonify({"error": "Request body required"}), 400

            galaxy_id = data.get("galaxy_id")
            sparc_gr = 0
            if galaxy_id:
                gal = get_galaxy_by_id(galaxy_id)
                if not gal:
                    return jsonify({
                        "error": "Galaxy not found: %s" % galaxy_id
                    }), 404
                observations = gal.get("observations", [])
                accel_ratio = gal.get("accel", 1.0)
                galaxy_name = gal.get("name", galaxy_id)
                sparc_gr = gal.get("galactic_radius", 0)
            else:
                observations = data.get("observations", [])
                accel_ratio = float(data.get("accel_ratio", 1.0))
                galaxy_name = data.get("name", "Custom")
                sparc_gr = float(data.get("galactic_radius", 0))

            if not observations or len(observations) < 3:
                return jsonify({
                    "error": "At least 3 observations required"
                }), 400

            obs_r = []
            obs_v = []
            obs_err = []
            for o in observations:
                r = float(o.get("r", 0))
                v = float(o.get("v", 0))
                if r > 0 and v > 0:
                    obs_r.append(r)
                    obs_v.append(v)
                    obs_err.append(max(float(o.get("err", 5.0)), 1.0))

            if len(obs_r) < 3:
                return jsonify({
                    "error": "At least 3 valid observations required"
                }), 400

            a0_eff = A0 * accel_ratio
            num_points = int(data.get("num_points", 500))

            # Chart data extends 1% past the horizon so the curve
            # always reaches beyond the R_env line on the graph.
            chart_max = float(data.get("chart_max_r", 0))
            if chart_max <= 0:
                chart_max = max(
                    sparc_gr * 1.15 if sparc_gr else 0,
                    max(obs_r) * 1.15)

            # Extract photometric scale lengths before Bayesian fit
            # so they can constrain the optimizer
            photo_scales = None
            if galaxy_id:
                mass_model = gal.get("mass_model", {})
                if mass_model:
                    photo_scales = {
                        "ab": mass_model.get("bulge", {}).get("a", 0),
                        "Rd": mass_model.get("disk", {}).get("Rd", 0),
                        "Rg": mass_model.get("gas", {}).get("Rd", 0),
                    }

            result = fit_gfd_bayesian(
                obs_r, obs_v, obs_err, a0_eff,
                num_points=num_points,
                chart_max_r=chart_max,
                photo_scales=photo_scales)
            result["galaxy"] = galaxy_name
            result["sparc_galactic_radius"] = sparc_gr

            # Photometric mass model GFD curve (if galaxy has mass_model)
            if galaxy_id:
                mass_model = gal.get("mass_model", {})
                if mass_model:
                    photometry = {
                        "Mb": mass_model.get("bulge", {}).get("M", 0),
                        "ab": mass_model.get("bulge", {}).get("a", 0),
                        "Md": mass_model.get("disk", {}).get("M", 0),
                        "Rd": mass_model.get("disk", {}).get("Rd", 0),
                        "Mg": mass_model.get("gas", {}).get("M", 0),
                        "Rg": mass_model.get("gas", {}).get("Rd", 0),
                    }
                    photo_result = derive_mass_parameters_from_photometry(
                        photometry, a0_eff)
                    if "error" not in photo_result:
                        pm = photo_result["mass_model"]
                        Mb_p = pm["bulge"]["M"]
                        ab_p = pm["bulge"]["a"]
                        Md_p = pm["disk"]["M"]
                        Rd_p = pm["disk"]["Rd"]
                        Mg_p = pm["gas"]["M"]
                        Rg_p = pm["gas"]["Rd"]

                        chart_radii = result.get("chart", {}).get(
                            "radii", [])
                        photo_vels = gfd_velocity_curve(
                            chart_radii,
                            Mb_p, ab_p, Md_p, Rd_p, Mg_p, Rg_p,
                            a0_eff)
                        result["chart"]["gfd_photometric"] = [
                            round(v, 4) for v in photo_vels]
                        result["photometric_mass_model"] = pm
                        result["photometric_M_total"] = photo_result[
                            "M_total"]

                        # R_vis from photometric model (actual observed
                        # mass distribution, not Bayesian)
                        photo_geom = solve_field_geometry(
                            Mb_p, ab_p, Md_p, Rd_p, Mg_p, Rg_p,
                            a0_eff)
                        r_vis_photo = photo_geom.get(
                            "visible_radius_kpc")
                        if r_vis_photo:
                            result["field_geometry"][
                                "visible_radius_kpc"] = r_vis_photo
                        r_vis_90 = photo_geom.get(
                            "visible_radius_90_kpc")
                        if r_vis_90:
                            result["field_geometry"][
                                "visible_radius_90_kpc"] = r_vis_90
                        r_vis_99 = photo_geom.get(
                            "visible_radius_99_kpc")
                        if r_vis_99:
                            result["field_geometry"][
                                "visible_radius_99_kpc"] = r_vis_99

                        # Extend chart radii if R_vis_99 exceeds
                        # the current range so curves fill the band
                        current_max = chart_radii[-1] if chart_radii else 0
                        r_vis_99_val = r_vis_99 or 0
                        needed_max = max(
                            current_max,
                            r_vis_99_val * 1.10,
                            (r_vis_90 or 0) * 1.10)
                        if needed_max > current_max * 1.05:
                            bay_p = result.get("mass_model", {})
                            bM = bay_p.get("bulge", {}).get("M", 0)
                            ba = bay_p.get("bulge", {}).get("a", 0)
                            dM = bay_p.get("disk", {}).get("M", 0)
                            dR = bay_p.get("disk", {}).get("Rd", 0)
                            gM = bay_p.get("gas", {}).get("M", 0)
                            gR = bay_p.get("gas", {}).get("Rd", 0)
                            n_pts = len(chart_radii) or 500
                            dr = needed_max / n_pts
                            chart_radii = [dr * (i + 1)
                                           for i in range(n_pts)]
                            base_new = gfd_velocity_curve(
                                chart_radii,
                                bM, ba, dM, dR, gM, gR,
                                a0_eff)
                            photo_vels = gfd_velocity_curve(
                                chart_radii,
                                Mb_p, ab_p, Md_p, Rd_p,
                                Mg_p, Rg_p, a0_eff)
                            result["chart"]["radii"] = [
                                round(r, 4) for r in chart_radii]
                            result["chart"]["gfd_base"] = [
                                round(v, 4) for v in base_new]
                            result["chart"]["gfd_photometric"] = [
                                round(v, 4) for v in photo_vels]

                        # Delta histogram: v^2_base - v^2_photo
                        # at chart radii (smooth curves)
                        base_vels = result["chart"].get("gfd_base", [])
                        delta_v2 = []
                        covariant_vels = []
                        for j in range(len(chart_radii)):
                            vb = base_vels[j] if j < len(base_vels) else 0
                            vp = photo_vels[j] if j < len(photo_vels) else 0
                            d = vb * vb - vp * vp
                            delta_v2.append(round(d, 2))
                            v_cov_sq = vp * vp + d
                            covariant_vels.append(
                                round(math.sqrt(max(v_cov_sq, 0)), 4))
                        result["chart"]["delta_v2"] = delta_v2
                        result["chart"]["gfd_covariant"] = covariant_vels

                        # Sigma diagnostic at observation points
                        bay_params = result.get("mass_model", {})
                        bp_b = bay_params.get("bulge", {})
                        bp_d = bay_params.get("disk", {})
                        bp_g = bay_params.get("gas", {})
                        obs_deltas = []
                        sum_delta_pos = 0.0
                        sum_delta_neg = 0.0
                        sum_v2_photo = 0.0
                        for j in range(len(obs_r)):
                            vb = gfd_velocity(
                                obs_r[j],
                                bp_b.get("M", 0), bp_b.get("a", 0),
                                bp_d.get("M", 0), bp_d.get("Rd", 0),
                                bp_g.get("M", 0), bp_g.get("Rd", 0),
                                a0_eff)
                            vp = gfd_velocity(
                                obs_r[j],
                                Mb_p, ab_p, Md_p, Rd_p, Mg_p, Rg_p,
                                a0_eff)
                            d = vb * vb - vp * vp
                            obs_deltas.append({
                                "r": obs_r[j],
                                "delta_v2": round(d, 1),
                                "v_base": round(vb, 1),
                                "v_photo": round(vp, 1),
                            })
                            if d > 0:
                                sum_delta_pos += d
                            else:
                                sum_delta_neg += d
                            sum_v2_photo += vp * vp

                        sigma_net = (
                            (sum_delta_pos + sum_delta_neg) / sum_v2_photo
                            if sum_v2_photo > 0 else 0)
                        result["vortex_signal"] = {
                            "obs_deltas": obs_deltas,
                            "sigma_net": round(sigma_net, 4),
                            "energy_boost": round(sum_delta_pos, 0),
                            "energy_suppress": round(sum_delta_neg, 0),
                            "energy_ratio": round(
                                abs(sum_delta_pos / sum_delta_neg)
                                if sum_delta_neg != 0 else 0, 2),
                        }

            return jsonify(result)

        @bp.route("/sandbox/compare_all", methods=["POST"])
        def sandbox_compare_all():
            """Run comparison across all catalog galaxies.

            No input required. Returns a summary table.
            """
            all_grouped = get_all_galaxies()
            all_galaxies = []
            for group in all_grouped.values():
                if isinstance(group, list):
                    all_galaxies.extend(group)
            num_points = 500
            results = []

            for gal in all_galaxies:
                gal_id = gal.get("id", "unknown")
                name = gal.get("name", gal_id).split("(")[0].strip()
                observations = gal.get("observations", [])
                mass_model = gal.get("mass_model", {})
                gr = gal.get("galactic_radius", 0)
                accel = gal.get("accel", 1.0)
                max_r = gal.get("distance", 30)

                if not observations or len(observations) < 3:
                    continue
                if not mass_model:
                    continue

                # Pure pipeline
                try:
                    pure = infer_from_observations(
                        observations, accel, num_points)
                except Exception as e:
                    pure = {"error": str(e)}

                # Existing pipeline
                try:
                    existing = optimize_inference(
                        mass_model, max_r, num_points,
                        observations, accel, float(gr))
                except Exception as e:
                    existing = {"error": str(e)}

                # Extract comparison metrics
                row = {"galaxy": name, "id": gal_id}

                if isinstance(pure, dict) and "mass_model" in pure:
                    pm = pure.get("mass_model", {})
                    pfg = pure.get("field_geometry", pure.get(
                        "field_geometry_bary", {}))
                    row["pure_M_total"] = round(
                        sum(c.get("M", 0) for c in pm.values()
                            if isinstance(c, dict)), 2)
                    row["pure_Rt"] = pfg.get("throat_radius_kpc")
                    row["pure_Renv"] = pfg.get(
                        "envelope_radius_kpc")
                    row["pure_throughput"] = pure.get("throughput")
                    row["pure_rms"] = pure.get("sigma_rms")
                    row["pure_f_gas"] = pure.get("f_gas")
                    row["pure_gfd_rms"] = pure.get("gfd_base_rms")
                    disp = pure.get("displacement")
                    if disp:
                        row["pure_Rt_obs"] = disp.get("Rt_obs")
                elif isinstance(pure, dict):
                    row["pure_error"] = pure.get("error", "unknown")

                if isinstance(existing, dict) and "mass_model" in existing:
                    em = existing["mass_model"]
                    efg = existing.get("field_geometry", {})
                    row["existing_M_total"] = round(
                        sum(c.get("M", 0)
                            for c in em.values()
                            if isinstance(c, dict)), 2)
                    row["existing_Rt"] = efg.get("throat_radius_kpc")
                    row["existing_Renv"] = efg.get(
                        "envelope_radius_kpc")
                    row["existing_throughput"] = existing.get(
                        "throughput")
                    row["existing_rms"] = existing.get("rms")
                    row["catalog_Renv"] = gr
                elif isinstance(existing, dict):
                    row["existing_error"] = existing.get(
                        "error", "unknown")

                results.append(row)

            return jsonify({"results": results, "count": len(results)})
