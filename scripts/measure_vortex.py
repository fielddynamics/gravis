#!/usr/bin/env python3
"""
measure_vortex.py
==================
Measure the vortex mass profile M_vortex(r) directly from the
difference between the Bayesian and photometric GFD curves.

Both curves use the same field equation x^2/(1+x) = yN.
The Bayesian curve has yN from effective mass (baryonic + vortex).
The photometric curve has yN from baryonic mass only.

The difference in yN at each radius gives the vortex contribution:
  yN_vortex(r) = yN_bayesian(r) - yN_photometric(r)
  M_vortex(<r) = yN_vortex(r) * a0 * r^2 / (G * M_SUN)

No fitting needed. The vortex field is fully determined by the
two curves that are already computed.
"""

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.galaxies import PREDICTION_GALAXIES
from physics.constants import G, M_SUN, KPC_TO_M, A0
from physics.services.sandbox.bayesian_fit import (
    _model_enc,
    _hernquist_enc,
    _disk_enc,
    derive_mass_parameters_from_photometry,
    fit_gfd_to_observations_with_bayesian,
    gfd_velocity,
)
from physics.services.rotation.inference import solve_field_geometry


def measure_vortex_profile(galaxy):
    gid = galaxy["id"]
    mm = galaxy["mass_model"]
    obs = galaxy["observations"]
    accel = galaxy.get("accel", 1.0)
    a0_eff = A0 * accel

    obs_r = [p["r"] for p in obs]
    obs_v = [p["v"] for p in obs]
    obs_err = [p["err"] for p in obs]

    # Step 1: Bayesian fit -> effective mass model (blue curve)
    bay = fit_gfd_to_observations_with_bayesian(
        obs_r, obs_v, obs_err, a0_eff)
    Mb_b, ab_b, Md_b, Rd_b, Mg_b, Rg_b = bay["params"]

    # Step 2: Photometric mass model (green curve)
    photometry = {
        "Mb": mm["bulge"]["M"], "ab": mm["bulge"]["a"],
        "Md": mm["disk"]["M"], "Rd": mm["disk"]["Rd"],
        "Mg": mm["gas"]["M"], "Rg": mm["gas"]["Rd"],
    }
    photo = derive_mass_parameters_from_photometry(photometry, a0_eff)
    if "error" in photo:
        return None

    pm = photo["mass_model"]
    Mb_p = pm["bulge"]["M"]
    ab_p = pm["bulge"]["a"]
    Md_p = pm["disk"]["M"]
    Rd_p = pm["disk"]["Rd"]
    Mg_p = pm["gas"]["M"]
    Rg_p = pm["gas"]["Rd"]

    topo = photo.get("topology", {})
    R_t = topo.get("r_t", 0)
    R_env = topo.get("r_env", 0)
    cycle = topo.get("cycle", 0)

    # Step 3: At each observation radius, compute both yN values
    # and derive M_vortex
    rows = []
    for r_kpc in obs_r:
        r_m = r_kpc * KPC_TO_M
        if r_m <= 0:
            continue

        # Bayesian (effective) enclosed mass
        M_enc_bay = _model_enc(r_kpc, Mb_b, ab_b, Md_b, Rd_b, Mg_b, Rg_b)
        # Photometric (baryonic) enclosed mass
        M_enc_photo = _model_enc(r_kpc, Mb_p, ab_p, Md_p, Rd_p, Mg_p, Rg_p)

        # yN values
        yN_bay = G * M_enc_bay * M_SUN / (r_m * r_m * a0_eff)
        yN_photo = G * M_enc_photo * M_SUN / (r_m * r_m * a0_eff)

        # Vortex contribution
        yN_vortex = yN_bay - yN_photo
        M_vortex = yN_vortex * a0_eff * r_m * r_m / (G * M_SUN)

        # Velocities
        v_bay = gfd_velocity(r_kpc, Mb_b, ab_b, Md_b, Rd_b, Mg_b, Rg_b, a0_eff)
        v_photo = gfd_velocity(r_kpc, Mb_p, ab_p, Md_p, Rd_p, Mg_p, Rg_p, a0_eff)

        # Fractional position in outer arm
        f = 0.0
        zone = "INNER"
        if R_t > 0 and R_env > R_t:
            if r_kpc >= R_t:
                f = min((r_kpc - R_t) / (R_env - R_t), 1.0)
                zone = "OUTER"
            else:
                f = -(R_t - r_kpc) / R_t
                zone = "INNER"

        rows.append({
            "r": r_kpc,
            "M_enc_bay": M_enc_bay,
            "M_enc_photo": M_enc_photo,
            "M_vortex": M_vortex,
            "yN_bay": yN_bay,
            "yN_photo": yN_photo,
            "yN_vortex": yN_vortex,
            "v_bay": v_bay,
            "v_photo": v_photo,
            "dv2": v_bay**2 - v_photo**2,
            "f": f,
            "zone": zone,
        })

    return {
        "id": gid,
        "R_t": R_t,
        "R_env": R_env,
        "cycle": cycle,
        "bay_params": bay["params"],
        "bay_rms": bay.get("rms", 0),
        "photo_params": [Mb_p, ab_p, Md_p, Rd_p, Mg_p, Rg_p],
        "M_total_bay": Mb_b + Md_b + Mg_b,
        "M_total_photo": Mb_p + Md_p + Mg_p,
        "rows": rows,
    }


def main():
    galaxies_to_analyze = ["m33", "milky_way", "ngc3198", "ngc2403"]

    for gid_target in galaxies_to_analyze:
        gal = None
        for g in PREDICTION_GALAXIES:
            if g["id"] == gid_target:
                gal = g
                break
        if gal is None:
            continue

        result = measure_vortex_profile(gal)
        if result is None:
            print("%s: SKIP (photometry failed)" % gid_target)
            continue

        rows = result["rows"]
        R_t = result["R_t"]
        R_env = result["R_env"]

        bar = "=" * 100
        print(bar)
        print("  %s  (Rt=%.1f, Renv=%.1f, cycle=%d)" % (
            gid_target.upper(), R_t, R_env, result["cycle"]))
        print("  Bayesian M_total = %.3e  Photometric M_total = %.3e" % (
            result["M_total_bay"], result["M_total_photo"]))
        print("  Bayesian params: Mb=%.2e ab=%.2f Md=%.2e Rd=%.2f Mg=%.2e Rg=%.2f" % tuple(result["bay_params"]))
        print("  Photo    params: Mb=%.2e ab=%.2f Md=%.2e Rd=%.2f Mg=%.2e Rg=%.2f" % tuple(result["photo_params"]))
        print(bar)
        print("  %5s  %5s  %8s  %8s  %10s  %8s  %8s  %8s  %6s" % (
            "r", "zone", "v_bay", "v_photo",
            "dv2", "M_vortex", "yN_bay", "yN_photo", "f"))
        print("  " + "-" * 96)

        for row in rows:
            m_v_str = "%+.2e" % row["M_vortex"]
            print("  %5.1f  %5s  %8.1f  %8.1f  %+10.0f  %8s  %8.5f  %8.5f  %+6.3f" % (
                row["r"], row["zone"],
                row["v_bay"], row["v_photo"],
                row["dv2"], m_v_str,
                row["yN_bay"], row["yN_photo"],
                row["f"]))

        # Summary: ratio of vortex yN to photometric yN in outer arm
        outer_rows = [r for r in rows if r["zone"] == "OUTER" and r["yN_photo"] > 0]
        if outer_rows:
            ratios = [r["yN_vortex"] / r["yN_photo"] for r in outer_rows]
            print()
            print("  OUTER ARM: yN_vortex / yN_photo ratios:")
            print("    min=%.4f  max=%.4f  mean=%.4f" % (
                min(ratios), max(ratios), sum(ratios)/len(ratios)))

            # Check if M_vortex grows monotonically
            m_vortex = [r["M_vortex"] for r in outer_rows]
            monotonic = all(m_vortex[i] <= m_vortex[i+1]
                          for i in range(len(m_vortex)-1))
            print("    M_vortex monotonic: %s" % monotonic)
            print("    M_vortex range: %.2e to %.2e" % (
                min(m_vortex), max(m_vortex)))

        print()


if __name__ == "__main__":
    main()
