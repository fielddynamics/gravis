#!/usr/bin/env python3
"""
sigma_survey.py
================
Run the full covariant sigma fit across all 22 SPARC galaxies.

For each galaxy:
  1. Photometric masses -> topology -> corrected M_total, M_gas, Mb, Md
  2. GFD base velocity (no vortex) -> RMS_base
  3. Fit sigma (1 free param) -> RMS_covariant
  4. Compare to 6-param Bayesian fit -> RMS_bayesian

Shows whether the single-parameter vortex correction from the
covariant action can match or beat the 6-parameter Bayesian fit.
"""

import math
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.galaxies import PREDICTION_GALAXIES
from physics.constants import A0
from physics.services.sandbox.bayesian_fit import (
    derive_mass_parameters_from_photometry,
    fit_sigma_from_photometric_model,
    gfd_velocity,
    fit_gfd_to_observations_with_bayesian,
)


def main():
    t0 = time.time()
    bar = "=" * 130
    sep = "-" * 130

    print(bar)
    print("  COVARIANT SIGMA FIT: 1 Free Parameter vs 6-Parameter Bayesian")
    print(bar)
    print()
    print("  For each galaxy:")
    print("    Photometric masses (fixed) -> GFD base (sigma=0) -> RMS_base")
    print("    Fit sigma from covariant action -> RMS_sigma")
    print("    Compare to 6-param Bayesian -> RMS_bay")
    print()
    print(sep)
    print("  %-12s %3s | %6s %6s | %8s %8s %8s | %6s %6s" % (
        "Galaxy", "Cyc", "R_t", "R_env",
        "RMS_base", "RMS_sig", "RMS_bay",
        "sigma", "improv"))
    print(sep)

    all_results = []

    for gal in PREDICTION_GALAXIES:
        gid = gal["id"]
        mm = gal["mass_model"]
        obs = gal["observations"]
        accel = gal.get("accel", 1.0)
        a0_eff = A0 * accel

        obs_r = [p["r"] for p in obs]
        obs_v = [p["v"] for p in obs]
        obs_err = [p["err"] for p in obs]

        t_gal = time.time()

        # Step 1: Photometric mass model
        photometry = {
            "Mb": mm["bulge"]["M"], "ab": mm["bulge"]["a"],
            "Md": mm["disk"]["M"], "Rd": mm["disk"]["Rd"],
            "Mg": mm["gas"]["M"], "Rg": mm["gas"]["Rd"],
        }
        photo = derive_mass_parameters_from_photometry(photometry, a0_eff)
        if "error" in photo:
            print("  %-12s  SKIP (%s)" % (gid, photo["error"]))
            continue

        pm = photo["mass_model"]
        topo = photo.get("topology", {})
        R_t = topo.get("r_t", 0)
        R_env = topo.get("r_env", 0)
        cycle = topo.get("cycle", 0)

        if R_t <= 0 or R_env <= 0:
            print("  %-12s  SKIP (no geometry)" % gid)
            continue

        # Step 2: Base RMS (no vortex)
        ss_base = 0.0
        for j in range(len(obs_r)):
            vp = gfd_velocity(obs_r[j],
                              pm["bulge"]["M"], pm["bulge"]["a"],
                              pm["disk"]["M"], pm["disk"]["Rd"],
                              pm["gas"]["M"], pm["gas"]["Rd"],
                              a0_eff)
            ss_base += (obs_v[j] - vp) ** 2
        rms_base = math.sqrt(ss_base / len(obs_r))

        # Step 3: Fit sigma
        sigma_result = fit_sigma_from_photometric_model(
            pm, obs_r, obs_v, obs_err, a0_eff, R_t, R_env)
        rms_sigma = sigma_result["rms"]
        sigma_val = sigma_result["sigma"]
        improvement = sigma_result["improvement"]

        # Step 4: Bayesian 6-param fit for comparison
        bay = fit_gfd_to_observations_with_bayesian(
            obs_r, obs_v, obs_err, a0_eff)
        rms_bay = bay.get("rms", 0)

        dt = time.time() - t_gal

        wins_bay = rms_sigma < rms_bay
        flag = " ** " if wins_bay else "    "

        all_results.append({
            "id": gid, "cycle": cycle,
            "R_t": R_t, "R_env": R_env,
            "rms_base": rms_base, "rms_sigma": rms_sigma,
            "rms_bay": rms_bay, "sigma": sigma_val,
            "improvement": improvement, "dt": dt,
        })

        print("  %-12s %3d | %6.1f %6.1f | %7.1f  %7.1f  %7.1f  | %+6.2f %+5.1f%%%s" % (
            gid, cycle, R_t, R_env,
            rms_base, rms_sigma, rms_bay,
            sigma_val, improvement, flag))

    elapsed = time.time() - t0

    # Summary
    print(sep)
    print()
    print(bar)
    print("  SUMMARY (%d galaxies, %.0f seconds)" % (len(all_results), elapsed))
    print(bar)
    print()

    def median(arr):
        s = sorted(arr)
        return s[len(s) // 2] if s else 0

    rms_bases = [r["rms_base"] for r in all_results]
    rms_sigmas = [r["rms_sigma"] for r in all_results]
    rms_bays = [r["rms_bay"] for r in all_results]
    improvements = [r["improvement"] for r in all_results]
    sigma_vals = [r["sigma"] for r in all_results]

    n_beats_bay = sum(1 for r in all_results if r["rms_sigma"] < r["rms_bay"])
    n_improves = sum(1 for r in all_results if r["improvement"] > 0)

    print("  RMS comparison (median across galaxies):")
    print("    Base (photometric, no vortex): %5.1f km/s" % median(rms_bases))
    print("    Covariant (1 param sigma):     %5.1f km/s" % median(rms_sigmas))
    print("    Bayesian (6 param fit):        %5.1f km/s" % median(rms_bays))
    print()
    print("  Sigma improves over base:      %d / %d galaxies" % (n_improves, len(all_results)))
    print("  Sigma beats 6-param Bayesian:  %d / %d galaxies  **" % (n_beats_bay, len(all_results)))
    print()
    print("  Sigma range: %+.2f to %+.2f" % (min(sigma_vals), max(sigma_vals)))
    print("  Improvement range: %+.1f%% to %+.1f%%" % (min(improvements), max(improvements)))
    print()
    print("  Interpretation:")
    print("    sigma > 0: gas-dominated, vortex pushes outward")
    print("    sigma < 0: stellar-dominated, vortex pulls inward")
    print("    sigma = 0: no vortex correction needed")


if __name__ == "__main__":
    main()
