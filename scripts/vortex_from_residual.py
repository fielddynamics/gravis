#!/usr/bin/env python3
"""
vortex_from_residual.py
========================
Extract the vortex power output (sigma) from the residual between
observations and the GFD base curve computed from topology-correct masses.

THE IDEA:
  Topology + photometry give the CORRECT baryonic masses (0.4% M_total).
  When plugged into the GFD base equation, this predicts v_base(r).
  The RESIDUAL between v_observed and v_base IS the vortex correction:

     v_obs(r) - v_base(r) = delta_v(r) = vortex signal

  From the covariant action with vortex core, this signal should be
  antisymmetric about R_t:
    - BELOW v_base inside R_t (suppression: field compressed)
    - ABOVE v_base outside R_t (enhancement: field expanded)
    - Zero at R_t (antisymmetric node)

  The amplitude of this signal is sigma (the throughput).

  This means we don't need to FIT sigma as part of a Bayesian
  optimization. We MEASURE it directly from the residual.
"""

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.galaxies import PREDICTION_GALAXIES
from physics.constants import A0, G, M_SUN, KPC_TO_M, THROAT_YN, HORIZON_YN
from physics.aqual import solve_x as aqual_solve_x
from physics.services.rotation.inference import solve_field_geometry
from physics.services.sandbox.bayesian_fit import gfd_velocity


def gfd_base_velocity(r_kpc, Mb, ab, Md, Rd, Mg, Rg, a0_eff):
    """GFD base velocity from the standard field equation (no vortex)."""
    return gfd_velocity(r_kpc, Mb, ab, Md, Rd, Mg, Rg, a0_eff)


def hernquist_frac(r, a):
    if r <= 0 or a <= 0:
        return 0.0
    return r * r / ((r + a) * (r + a))


def disk_frac(r, Rd):
    if r <= 0 or Rd <= 0:
        return 0.0
    x = r / Rd
    if x > 50:
        return 1.0
    return 1.0 - (1.0 + x) * math.exp(-x)


def solve_3x3(A, b):
    det = (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
         - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
         + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))
    if abs(det) < 1e-30:
        return None
    x0 = (b[0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
        - A[0][1] * (b[1] * A[2][2] - A[1][2] * b[2])
        + A[0][2] * (b[1] * A[2][1] - A[1][1] * b[2])) / det
    x1 = (A[0][0] * (b[1] * A[2][2] - A[1][2] * b[2])
        - b[0] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
        + A[0][2] * (A[1][0] * b[2] - b[1] * A[2][0])) / det
    x2 = (A[0][0] * (A[1][1] * b[2] - b[1] * A[2][1])
        - A[0][1] * (A[1][0] * b[2] - b[1] * A[2][0])
        + b[0] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])) / det
    return (x0, x1, x2)


def topology_masses(Mb_ph, ab_ph, Md_ph, Rd_ph, Mg_ph, Rg_ph, a0_eff):
    """Get topology-corrected masses + field geometry."""
    geom = solve_field_geometry(Mb_ph, ab_ph, Md_ph, Rd_ph, Mg_ph, Rg_ph, a0_eff)
    r_env = geom.get("envelope_radius_kpc")
    r_t = geom.get("throat_radius_kpc")
    cycle = geom.get("cycle", 3)
    yN_at_rt = geom.get("yN_at_throat", 0.0)

    if not r_env or not r_t or r_env <= 0 or r_t <= 0:
        return None

    r_env_m = r_env * KPC_TO_M
    M_horizon = HORIZON_YN * r_env_m**2 * a0_eff / (G * M_SUN)

    if cycle == 3:
        r_t_m = r_t * KPC_TO_M
        M_throat = THROAT_YN * r_t_m**2 * a0_eff / (G * M_SUN)
    else:
        r_t_m = r_t * KPC_TO_M
        M_throat = yN_at_rt * r_t_m**2 * a0_eff / (G * M_SUN)

    fb_env = hernquist_frac(r_env, ab_ph)
    fd_env = disk_frac(r_env, Rd_ph)
    fg_env = disk_frac(r_env, Rg_ph)
    fb_t = hernquist_frac(r_t, ab_ph)
    fd_t = disk_frac(r_t, Rd_ph)
    fg_t = disk_frac(r_t, Rg_ph)

    ratio = 2.0
    sol = None
    for _ in range(30):
        A = [[fb_env, fd_env, fg_env],
             [fb_t,   fd_t,   fg_t],
             [0.0,    1.0,    -ratio]]
        b = [M_horizon, M_throat, 0.0]
        sol = solve_3x3(A, b)
        if sol is None:
            return None
        Mb, Md, Mg = sol
        Mt = Mb + Md + Mg
        if Mt <= 0:
            break
        fg = max(Mg, 0) / Mt
        fg = max(0.01, min(fg, 0.99))
        fd_pred = -0.826 * fg + 0.863
        fd_pred = max(0.02, min(fd_pred, 0.95))
        new_ratio = fd_pred / fg
        new_ratio = max(0.01, min(new_ratio, 20.0))
        if abs(new_ratio - ratio) < 0.001:
            break
        ratio = ratio * 0.5 + new_ratio * 0.5

    if sol is None:
        return None

    Mb_t, Md_t, Mg_t = sol
    photo_ratio = Mb_ph / Md_ph if Md_ph > 0 else 0
    M_total = Mb_t + Md_t + Mg_t
    M_gas = max(Mg_t, 0)
    M_stellar = M_total - M_gas
    if M_stellar > 0 and photo_ratio > 0:
        M_bulge = M_stellar * photo_ratio / (1 + photo_ratio)
        M_disk = M_stellar - M_bulge
    else:
        M_bulge = max(Mb_t, 0)
        M_disk = max(Md_t, 0)

    return {
        "Mb": max(M_bulge, 0), "Md": max(M_disk, 0), "Mg": max(M_gas, 0),
        "r_env": r_env, "r_t": r_t, "cycle": cycle,
    }


def main():
    bar = "=" * 130
    sep = "-" * 130

    print(bar)
    print("  VORTEX EXTRACTION: Measuring Sigma from the Residual")
    print(bar)
    print()
    print("  The GFD base equation with topology-correct masses predicts v_base(r).")
    print("  The residual  delta_v = v_obs - v_base  IS the vortex signal.")
    print("  If the vortex term is real, delta_v should be:")
    print("    Negative inside R_t  (field compressed by the vortex core)")
    print("    Zero at R_t          (antisymmetric node)")
    print("    Positive outside R_t (field expanded beyond the throat)")
    print()
    print(sep)

    all_results = []

    for gal in PREDICTION_GALAXIES:
        gid = gal["id"]
        mm = gal["mass_model"]
        obs = gal["observations"]
        accel = gal.get("accel", 1.0)
        a0_eff = A0 * accel

        Mb_s = mm["bulge"]["M"]
        ab_s = mm["bulge"]["a"]
        Md_s = mm["disk"]["M"]
        Rd_s = mm["disk"]["Rd"]
        Mg_s = mm["gas"]["M"]
        Rg_s = mm["gas"]["Rd"]

        # Get topology-corrected masses
        topo = topology_masses(Mb_s, ab_s, Md_s, Rd_s, Mg_s, Rg_s, a0_eff)
        if topo is None:
            print("  %-12s  SKIP (topology failed)" % gid)
            continue

        Mb = topo["Mb"]
        Md = topo["Md"]
        Mg = topo["Mg"]
        r_t = topo["r_t"]
        r_env = topo["r_env"]
        cycle = topo["cycle"]

        # Compute residuals at each observation point
        residuals_inner = []   # r < R_t
        residuals_outer = []   # R_t < r < R_env
        residuals_beyond = []  # r > R_env
        all_resid = []

        for p in obs:
            r = p["r"]
            v_obs = p["v"]
            v_base = gfd_base_velocity(r, Mb, ab_s, Md, Rd_s, Mg, Rg_s, a0_eff)
            delta_v = v_obs - v_base
            frac_delta = delta_v / v_obs if v_obs > 0 else 0

            all_resid.append((r, delta_v, frac_delta, v_obs, v_base))

            if r < r_t:
                residuals_inner.append(delta_v)
            elif r < r_env:
                residuals_outer.append(delta_v)
            else:
                residuals_beyond.append(delta_v)

        # Analyze the residual pattern
        mean_inner = sum(residuals_inner) / len(residuals_inner) if residuals_inner else 0
        mean_outer = sum(residuals_outer) / len(residuals_outer) if residuals_outer else 0
        mean_beyond = sum(residuals_beyond) / len(residuals_beyond) if residuals_beyond else 0

        # Is the pattern antisymmetric? (negative inside, positive outside)
        is_antisymmetric = mean_inner < 0 and mean_outer > 0

        # RMS of all residuals (this is the "work" being done)
        ss = sum(dv * dv for _, dv, _, _, _ in all_resid)
        rms = math.sqrt(ss / len(all_resid)) if all_resid else 0

        # Fractional RMS (relative to typical velocity)
        v_typical = sum(vo for _, _, _, vo, _ in all_resid) / len(all_resid) if all_resid else 1
        frac_rms = rms / v_typical * 100

        # Estimate sigma amplitude from the asymmetry
        # sigma ~ (mean_outer - mean_inner) / (2 * v_typical)
        sigma_est = (mean_outer - mean_inner) / (2 * v_typical) if v_typical > 0 else 0

        result = {
            "id": gid, "cycle": cycle,
            "r_t": r_t, "r_env": r_env,
            "mean_inner": mean_inner, "mean_outer": mean_outer,
            "mean_beyond": mean_beyond,
            "antisym": is_antisymmetric,
            "rms": rms, "frac_rms": frac_rms,
            "sigma_est": sigma_est,
            "n_inner": len(residuals_inner),
            "n_outer": len(residuals_outer),
            "n_beyond": len(residuals_beyond),
            "resid": all_resid,
        }
        all_results.append(result)

        sym_flag = "YES" if is_antisymmetric else " no"
        print("  %-12s cyc=%d R_t=%5.1f R_env=%6.1f | "
              "inner=%+6.1f (%d pts) outer=%+6.1f (%d pts) beyond=%+5.1f (%d) | "
              "antisym=%s RMS=%5.1f (%4.1f%%) sigma_est=%+.3f" % (
            gid, cycle, r_t, r_env,
            mean_inner, len(residuals_inner),
            mean_outer, len(residuals_outer),
            mean_beyond, len(residuals_beyond),
            sym_flag, rms, frac_rms, sigma_est))

    # ====================================================================
    #  SUMMARY
    # ====================================================================
    print(sep)
    print()
    print(bar)
    print("  SUMMARY")
    print(bar)
    print()

    n_antisym = sum(1 for r in all_results if r["antisym"])
    n_total = len(all_results)
    print("  Antisymmetric pattern (negative inside R_t, positive outside):")
    print("    %d / %d galaxies (%.0f%%)" % (n_antisym, n_total, 100 * n_antisym / n_total))
    print()

    # Sort by sigma estimate
    sorted_results = sorted(all_results, key=lambda r: r["sigma_est"])
    print("  %-12s %3s | %7s | %7s | %7s | %5s" % (
        "Galaxy", "Cyc", "sigma", "RMS", "fRMS", "Asym"))
    print("  " + "-" * 60)
    for r in sorted_results:
        sym = "YES" if r["antisym"] else " no"
        print("  %-12s %3d | %+7.3f | %7.1f | %5.1f%% | %s" % (
            r["id"], r["cycle"], r["sigma_est"],
            r["rms"], r["frac_rms"], sym))

    print()
    sigma_vals = [r["sigma_est"] for r in all_results]
    rms_vals = [r["rms"] for r in all_results]
    frac_rms_vals = [r["frac_rms"] for r in all_results]
    print("  Sigma range:    %+.3f to %+.3f" % (min(sigma_vals), max(sigma_vals)))
    print("  RMS range:      %.1f to %.1f km/s" % (min(rms_vals), max(rms_vals)))
    print("  Fractional RMS: %.1f%% to %.1f%%" % (min(frac_rms_vals), max(frac_rms_vals)))

    # Detailed per-galaxy residual structure for a few interesting cases
    print()
    print(bar)
    print("  DETAILED RESIDUAL PROFILES (first 5 galaxies)")
    print(bar)
    for r in all_results[:5]:
        print()
        print("  %s (cycle=%d, R_t=%.1f, R_env=%.1f, sigma=%.3f)" % (
            r["id"], r["cycle"], r["r_t"], r["r_env"], r["sigma_est"]))
        print("    %6s %7s %7s %7s %6s %s" % ("r", "v_obs", "v_base", "delta", "%", "zone"))
        print("    " + "-" * 55)
        for rad, dv, fdv, vo, vb in r["resid"]:
            zone = "INNER" if rad < r["r_t"] else ("OUTER" if rad < r["r_env"] else "BEYOND")
            print("    %6.1f %7.1f %7.1f %+7.1f %+5.1f%% %s" % (
                rad, vo, vb, dv, fdv * 100, zone))


if __name__ == "__main__":
    main()
