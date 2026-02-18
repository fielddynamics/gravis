#!/usr/bin/env python3
"""
vortex_macd.py
===============
MACD-style vortex signal extraction.

Fast signal = v^2_base(r)       (GFD Bayesian fit, responds to observations)
Slow signal = v^2_photometric(r) (GFD from baryonic mass model only)
Delta       = Fast - Slow        (the vortex contribution at each radius)

sigma is extracted by fitting the delta profile to the topological
vortex shape. If the action is correct, sigma is constant across
all outer arm radii (zero scatter).
"""

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.galaxies import PREDICTION_GALAXIES
from physics.constants import G, M_SUN, KPC_TO_M, A0
from physics.aqual import solve_x as aqual_solve_x
from physics.services.sandbox.bayesian_fit import (
    _model_enc,
    derive_mass_parameters_from_photometry,
    fit_gfd_to_observations_with_bayesian,
    gfd_velocity,
)


def compute_vortex_signal(galaxy):
    """Compute the MACD-style vortex delta at each observation point."""
    gid = galaxy["id"]
    mm = galaxy["mass_model"]
    obs = galaxy["observations"]
    accel = galaxy.get("accel", 1.0)
    a0_eff = A0 * accel

    obs_r = [p["r"] for p in obs]
    obs_v = [p["v"] for p in obs]
    obs_err = [p["err"] for p in obs]

    # Bayesian fit (fast signal)
    bay = fit_gfd_to_observations_with_bayesian(
        obs_r, obs_v, obs_err, a0_eff)
    bay_params = bay["params"]

    # Photometric mass model (slow signal)
    photometry = {
        "Mb": mm["bulge"]["M"], "ab": mm["bulge"]["a"],
        "Md": mm["disk"]["M"], "Rd": mm["disk"]["Rd"],
        "Mg": mm["gas"]["M"], "Rg": mm["gas"]["Rd"],
    }
    photo = derive_mass_parameters_from_photometry(photometry, a0_eff)
    if "error" in photo:
        return None

    pm = photo["mass_model"]
    photo_params = [
        pm["bulge"]["M"], pm["bulge"]["a"],
        pm["disk"]["M"], pm["disk"]["Rd"],
        pm["gas"]["M"], pm["gas"]["Rd"],
    ]

    topo = photo.get("topology", {})
    R_t = topo.get("r_t", 0)
    R_env = topo.get("r_env", 0)
    L = R_env - R_t if R_env > R_t else 1.0

    # At each observation point, compute both v^2 values
    rows = []
    for i in range(len(obs_r)):
        r = obs_r[i]
        r_m = r * KPC_TO_M
        if r_m <= 0:
            continue

        v_base = gfd_velocity(r, *bay_params, a0_eff)
        v_photo = gfd_velocity(r, *photo_params, a0_eff)

        # The delta: v^2_base - v^2_photo (in km^2/s^2)
        delta_v2 = v_base ** 2 - v_photo ** 2

        # Fractional position in outer arm
        zone = "INNER" if r < R_t else "OUTER"
        f = max(0.0, min((r - R_t) / L, 1.0)) if r >= R_t else 0.0

        # Topological shape prediction for the outer arm:
        # vortex_shape = (4/13) * gN_photo * f^(3/4) * r  [in km^2/s^2]
        enc_photo = _model_enc(r, *photo_params)
        gN_photo = G * enc_photo * M_SUN / (r_m * r_m) if enc_photo > 0 else 0
        shape = 0.0
        if f > 0 and gN_photo > 0:
            shape = (4.0 / 13.0) * gN_photo * (f ** 0.75) * r_m / 1e6

        rows.append({
            "r": r,
            "v_obs": obs_v[i],
            "v_base": v_base,
            "v_photo": v_photo,
            "delta_v2": delta_v2,
            "zone": zone,
            "f": f,
            "shape": shape,
            "sigma_local": delta_v2 / shape if abs(shape) > 1e-10 else None,
        })

    return {
        "id": gid,
        "R_t": R_t,
        "R_env": R_env,
        "bay_rms": bay.get("rms", 0),
        "rows": rows,
    }


def main():
    galaxies = ["m33", "milky_way", "ngc2403", "ngc3198", "ngc5055", "ddo154"]

    for gid_target in galaxies:
        gal = None
        for g in PREDICTION_GALAXIES:
            if g["id"] == gid_target:
                gal = g
                break
        if gal is None:
            continue

        result = compute_vortex_signal(gal)
        if result is None:
            print("%s: SKIP" % gid_target)
            continue

        rows = result["rows"]
        R_t = result["R_t"]
        R_env = result["R_env"]

        bar = "=" * 105
        print(bar)
        print("  %s   Rt=%.1f  Renv=%.1f  (Bayesian RMS=%.1f km/s)" % (
            gid_target.upper(), R_t, R_env, result["bay_rms"]))
        print(bar)
        print("  %5s %5s  %7s %7s %7s | %9s %7s | %8s" % (
            "r", "zone", "v_obs", "v_base", "v_photo",
            "delta_v2", "shape", "sigma_i"))
        print("  " + "-" * 101)

        for row in rows:
            sig_str = "%8.2f" % row["sigma_local"] if row["sigma_local"] is not None else "     n/a"
            print("  %5.1f %5s  %7.1f %7.1f %7.1f | %+9.0f %7.1f | %s" % (
                row["r"], row["zone"],
                row["v_obs"], row["v_base"], row["v_photo"],
                row["delta_v2"], row["shape"],
                sig_str))

        # Extract sigma from outer arm points
        outer = [r for r in rows if r["zone"] == "OUTER"
                 and r["sigma_local"] is not None
                 and abs(r["shape"]) > 1e-10]
        if outer:
            sigmas = [r["sigma_local"] for r in outer]
            sigma_mean = sum(sigmas) / len(sigmas)
            sigma_std = math.sqrt(
                sum((s - sigma_mean) ** 2 for s in sigmas) / len(sigmas))
            cv = sigma_std / abs(sigma_mean) * 100 if sigma_mean != 0 else 999

            print()
            print("  SIGMA from outer arm (%d points):" % len(outer))
            print("    mean  = %.3f" % sigma_mean)
            print("    stdev = %.3f" % sigma_std)
            print("    CV    = %.1f%% (lower = more consistent)" % cv)
            print("    range = %.3f to %.3f" % (min(sigmas), max(sigmas)))

            # Show what the covariant curve would look like
            print()
            print("  COVARIANT CURVE (v_photo + sigma*shape):")
            print("  %5s  %7s %7s %7s %7s  %6s" % (
                "r", "v_obs", "v_base", "v_photo", "v_cov", "resid"))
            for row in rows:
                v_cov_sq = row["v_photo"]**2 + sigma_mean * row["shape"]
                v_cov = math.sqrt(max(v_cov_sq, 0))
                resid = row["v_obs"] - v_cov
                print("  %5.1f  %7.1f %7.1f %7.1f %7.1f  %+6.1f" % (
                    row["r"], row["v_obs"], row["v_base"],
                    row["v_photo"], v_cov, resid))

            # RMS of covariant vs base
            ss_cov = 0.0
            ss_base = 0.0
            for row in rows:
                v_cov_sq = row["v_photo"]**2 + sigma_mean * row["shape"]
                v_cov = math.sqrt(max(v_cov_sq, 0))
                ss_cov += (row["v_obs"] - v_cov) ** 2
                ss_base += (row["v_obs"] - row["v_base"]) ** 2
            rms_cov = math.sqrt(ss_cov / len(rows))
            rms_base = math.sqrt(ss_base / len(rows))
            print()
            print("  RMS: Bayesian=%.1f  Covariant(sigma=%.2f)=%.1f km/s" % (
                rms_base, sigma_mean, rms_cov))

        print()


if __name__ == "__main__":
    main()
